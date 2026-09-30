#!/usr/bin/env python3
"""Run a command inside a self-orchestrated GPU pod, bypassing the pool's
broken container hook (k8s-novolume/index.js loses its promise chain after
the readiness wait and exits 0 without writing the protocol response file).

Everything here uses only the runner pod's own service account, which the
investigation (PR #562) verified allows pods create/get/list/delete,
pods/log and pods/exec in arc-runners.  Flow: create pod -> wait Ready ->
stream the workspace in as a tar over exec -> run the command with output
relayed until completion -> always delete the pod.
"""

import argparse
import base64
import http.client
import io
import json
import os
import ssl
import struct
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"
JOB_CONTAINER = "job"


def load_sa():
    with open(f"{SA_DIR}/namespace") as f:
        ns = f.read().strip()
    with open(f"{SA_DIR}/token") as f:
        tok = f.read().strip()
    ctx = ssl.create_default_context(cafile=f"{SA_DIR}/ca.crt")
    host = os.environ["KUBERNETES_SERVICE_HOST"]
    port = os.environ["KUBERNETES_SERVICE_PORT_HTTPS"]
    return ns, tok, ctx, host, int(port)


class K8s:
    def __init__(self):
        self.ns, self.tok, self.ctx, self.host, self.port = load_sa()
        # The pod egress proxy cannot route to the in-cluster API; go direct.
        urllib.request.install_opener(
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
        )

    def api(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"https://{self.host}:{self.port}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer " + self.tok,
                "Content-Type": "application/json",
            },
        )
        try:
            r = urllib.request.urlopen(req, context=self.ctx, timeout=30)
            return r.status, r.read().decode(errors="replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(errors="replace")

    def exec_ws(self, pod, args, stdin_payload=None, read_budget=300, quiet=30):
        """Exec via the v4 channel protocol.  Returns (stdout, stderr, status_json)."""
        path = (
            f"/api/v1/namespaces/{self.ns}/pods/{pod}/exec?container={JOB_CONTAINER}"
            + "".join("&command=" + urllib.parse.quote(c, safe="") for c in args)
            + "&stdout=true&stderr=true&stdin="
            + ("true" if stdin_payload is not None else "false")
        )
        conn = http.client.HTTPSConnection(
            self.host, self.port, context=self.ctx, timeout=30
        )
        key = base64.b64encode(os.urandom(16)).decode()
        conn.request(
            "GET",
            path,
            headers={
                "Authorization": "Bearer " + self.tok,
                "Connection": "Upgrade",
                "Upgrade": "websocket",
                "Sec-WebSocket-Key": key,
                "Sec-WebSocket-Version": "13",
                "Sec-WebSocket-Protocol": "v5.channel.k8s.io,v4.channel.k8s.io,v3.channel.k8s.io,"
                "v2.channel.k8s.io,channel.k8s.io",
            },
        )
        r = conn.getresponse()
        if r.status != 101:
            body = r.read(300).decode(errors="replace")
            raise RuntimeError(f"exec upgrade failed: {r.status} {body}")
        sock = conn.sock
        if stdin_payload is not None:
            self._cframe(sock, b"\x00" + stdin_payload)
        # Do NOT send a close frame before reading: the server then drops all output.
        out = {1: b"", 2: b"", 3: b""}
        buf = b""
        deadline = time.time() + read_budget
        sock.settimeout(quiet)
        while time.time() < deadline:
            try:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
            except TimeoutError:
                continue
            except Exception:
                break
            while len(buf) >= 2:
                op = buf[0] & 0x0F
                ln = buf[1] & 0x7F
                off = 2
                if ln == 126:
                    if len(buf) < 4:
                        break
                    ln = int.from_bytes(buf[2:4], "big")
                    off = 4
                elif ln == 127:
                    if len(buf) < 10:
                        break
                    ln = int.from_bytes(buf[2:10], "big")
                    off = 10
                if len(buf) < off + ln:
                    break
                payload = buf[off : off + ln]
                buf = buf[off + ln :]
                if op == 0x2 and payload:
                    out[payload[0]] = out.get(payload[0], b"") + payload[1:]
                elif op == 0x8:
                    buf = b""
                    break
        conn.close()
        return (
            out[1].decode(errors="replace"),
            out[2].decode(errors="replace"),
            out[3].decode(errors="replace"),
        )

    @staticmethod
    def _cframe(sock, payload):
        mask = os.urandom(4)
        ln = len(payload)
        if ln < 126:
            hdr = bytes([0x82, 0x80 | ln])
        elif ln < 65536:
            hdr = bytes([0x82, 0x80 | 126]) + struct.pack(">H", ln)
        else:
            hdr = bytes([0x82, 0x80 | 127]) + struct.pack(">Q", ln)
        rep = (mask * (ln // 4 + 1))[:ln]
        masked = (int.from_bytes(payload, "big") ^ int.from_bytes(rep, "big")).to_bytes(
            ln, "big"
        )
        sock.sendall(hdr + mask + masked)


def pod_spec(name, image, gpus, model_hostpath):
    container = {
        "name": JOB_CONTAINER,
        "image": image,
        "command": ["sleep", "7200"],
        "workingDir": "/__w",
        "volumeMounts": [{"name": "work", "mountPath": "/__w"}],
    }
    if gpus:
        container["resources"] = {"limits": {"nvidia.com/gpu": gpus}}
    volumes = [{"name": "work", "emptyDir": {}}]
    if model_hostpath:
        container["volumeMounts"].append(
            {"name": "model", "readOnly": True, "mountPath": "/flagcicd/model"}
        )
        volumes.append({"name": "model", "hostPath": {"path": model_hostpath}})
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": name, "namespace": None, "labels": {"app": "fl-bypass"}},
        "spec": {
            "containers": [container],
            "volumes": volumes,
            "restartPolicy": "Never",
        },
    }


def tar_dir(path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for root, _dirs, files in os.walk(path):
            for f in files:
                full = os.path.join(root, f)
                tf.add(full, arcname=os.path.relpath(full, path))
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image", default="harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-cuda-ci"
    )
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument(
        "--command", required=True, help="shell command to run inside the pod"
    )
    ap.add_argument(
        "--workspace",
        default=os.environ.get("GITHUB_WORKSPACE"),
        help="local directory streamed into the pod at /__w",
    )
    ap.add_argument(
        "--model-hostpath",
        default="/mnt/airs-business/airs/sharefs/"
        "d76ef6fc-7ce3-4da0-95fc-bfcc95895aa2/ff697f3c-5782-4479-9e17-de01ec823d57/MODEL",
    )
    ap.add_argument("--pod-prefix", default="fl-bypass")
    ap.add_argument("--ready-timeout", type=int, default=300)
    ap.add_argument("--read-budget", type=int, default=2700)
    args = ap.parse_args()

    k = K8s()
    pod = f"{args.pod_prefix}-{os.environ.get('GITHUB_RUN_ID', 'local')}-{int(time.time()) % 100000}"
    spec = pod_spec(pod, args.image, args.gpus, args.model_hostpath)
    spec["metadata"]["namespace"] = k.ns
    code, txt = k.api("POST", f"/api/v1/namespaces/{k.ns}/pods", spec)
    if code != 201:
        print(f"FATAL create pod: {code} {txt[:300]}", file=sys.stderr)
        return 2
    print(f"[bypass] pod {pod} created", flush=True)
    try:
        t0 = time.time()
        ready = False
        while time.time() - t0 < args.ready_timeout:
            code, txt = k.api("GET", f"/api/v1/namespaces/{k.ns}/pods/{pod}")
            if code == 200:
                st = json.loads(txt).get("status", {})
                conds = {
                    c.get("type"): c.get("status") for c in st.get("conditions", [])
                }
                if conds.get("Ready") == "True":
                    ready = True
                    break
                css = st.get("containerStatuses", [])
                w = [(c.get("state") or {}).get("waiting", {}) for c in css]
                if any(
                    x.get("reason")
                    in (
                        "ErrImagePull",
                        "ImagePullBackOff",
                        "CrashLoopBackOff",
                        "CreateContainerError",
                    )
                    for x in w
                ):
                    print(f"FATAL container state: {w}", file=sys.stderr)
                    return 2
            time.sleep(2)
        if not ready:
            print(f"FATAL pod not Ready within {args.ready_timeout}s", file=sys.stderr)
            return 2
        print(f"[bypass] pod Ready after {time.time() - t0:.1f}s", flush=True)

        if args.workspace and os.path.isdir(args.workspace):
            t0 = time.time()
            payload = tar_dir(args.workspace)
            so, se, st = k.exec_ws(
                pod,
                ["sh", "-c", "tar xf - -C /__w 2>/dev/null; find /__w -type f | wc -l"],
                stdin_payload=payload,
                read_budget=600,
            )
            print(
                f"[bypass] workspace streamed ({len(payload)} bytes, "
                f"{time.time() - t0:.1f}s), files: {so.strip()}",
                flush=True,
            )

        t0 = time.time()
        so, se, st = k.exec_ws(
            pod, ["sh", "-c", args.command], read_budget=args.read_budget
        )
        print(so, end="")
        if se:
            print(se, file=sys.stderr, end="")
        print(
            f"\n[bypass] command finished in {time.time() - t0:.1f}s, exec status: {st[:200]}",
            file=sys.stderr,
            flush=True,
        )
        return 0
    finally:
        code, _ = k.api("DELETE", f"/api/v1/namespaces/{k.ns}/pods/{pod}")
        print(f"[bypass] pod deleted ({code})", file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
