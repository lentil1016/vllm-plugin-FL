# MUSA (S5000) vLLM 0.24 适配问题调查报告

- 首版：2026-09-09 ～ 2026-09-10；**v2 修订：2026-09-15**（补全所有问题的原始报错日志摘录、逐步现象描述与精确复现步骤；修正证据保存位置；补充宿主机状态复查结论）
- 复现环境（逐项可核对）：
  - 宿主机 `bm-mthreads-bjsjq-zone1-moer-s5000-80g-38-23`，MTT S5000 ×8（每卡 81920MiB），mthreads-gmi 2.3.2 / Driver 3.3.5-server，Ubuntu 22.04（容器内 python3.10.12）
  - **宿主机自 2026-06-27 22:54 起未重启**（`uptime -s` 实测），登录横幅持续提示 `*** System restart required ***`
  - 容器 `harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.24.0-musa-ci`
    （vLLM 0.24.0+empty / torch 2.9.0 / torch_musa 2.9.0 / FlagGems 5.3.2.post1.dev22+gb1f939eb5 / flagtree 0.6.0+mthreads3.6）
  - 插件侧：flagos-ai/vllm-plugin-FL，PR #457（本报告随该 PR 交付，报告本身不含任何运行时 py 改动）
- 证据文件：宿主机 `/root/fl-musa-evidence/`（14 个文件，2026-09-15 经 `docker cp` 从容器 `musa-024-e2e` 导出；**首版报告"已备份宿主机"的表述有误**——当时的备份命令实际把文件留在了容器内，现已真正导出并双份保存）；复现脚本：宿主机 `/root/scripts_debug/`、`/root/tp_test.py`（均从容器导出）
- 结论速览：**五个故障模式，根因均不在插件层；①③④⑤归属摩尔（torch_musa/驱动），②一半归属 vLLM 上游、一半归属摩尔**。插件层无法根修，只能按 §0 的映射显式收缩测试面并等待根修。

---

## 0. 因问题被跳过/收缩的用例清单（与问题编号一一对应）

| 被跳过/收缩的用例或能力 | 原始形态 | 阻塞问题 | 恢复条件 |
|---|---|---|---|
| e2e inference `qwen3_6/27b_tp4_eager`（vendor 原有用例） | TP4 多卡推理 | ① | 摩尔修复 TP≥2 路径后恢复 |
| e2e serving `qwen3_6/35b_a3b_tp4_eager`（vendor 原有用例） | TP4 serving | ①＋② | ②修复可先恢复 serving 框架，①修复后恢复 TP4 |
| e2e serving 任务整体 | `tests/platforms/musa.yaml` serving 段 | ②（`vllm serve` 无法启动） | vLLM 上游或 torch_musa 任一方修复 fork 冲突 |
| benchmark（serve smoke 三件套） | `benchmark.enabled: true` | ②（`vllm bench serve` 同样要起服务） | 同上 |
| `enforce_eager=False` 变体 | `06b_tp1.yaml` parametrize `[true, false]` | ③（图模式捕获挂死） | 摩尔修复捕获路径后恢复 `[true, false]` |
| 多卡矩阵（现仅保留 `qwen3/06b_tp1` 单卡） | 原 TP4 矩阵 | ①（⑤在诊断中亦拦路，可用 `disable_custom_all_reduce=true` 绕） | 同① |

> 设计说明：以上收缩全部是**显式**的——`tests/platforms/musa.yaml` 内注释逐条标注问题编号，PR 描述与本报告互相引用；没有任何无痕跳过。当前保留的测试面（unit + functional + TP1 eager 推理 e2e）全部真实执行且通过（09-14 CI run 34815101547 全绿）。

## 1. 问题清单与时间线

| # | 故障 | 首次观测 | 复现性 | 09-15 复查 |
|---|---|---|---|---|
| ① | TP≥2 权重加载挂死（rank≥1 用户态自旋） | 09-09 | 09-09 必现；09-10 漂移为④⑤（见 §6） | 未复测（建议按 §2 步骤在重启后的宿主机复测） |
| ② | `vllm serve` 启动失败：fork 冲突 | 09-09 | 稳定复现（3 秒内退出） | 机制未变（版本未升级） |
| ③ | 图模式（torch.compile + PIECEWISE cudagraph）捕获挂死 | 09-10 CI | 干净 runner 必现（60 分钟零输出直至超时） | 未复测 |
| ④ | TP2 `profile_run` dummy 前向 **illegal memory access** | 09-10 | 当日两次复现，每次崩溃后泄漏 ~70GB 显存 | 设备现已干净（见 §5 修正） |
| ⑤ | init 期 `can_device_access_peer` 断言 | 09-10 | 状态依赖 | 未复测 |

共通点：**TP1（单卡、in-process、eager）始终健康**——同一容器、同一模型、同一天内 TP1-OK 与 TP2 故障并存。

**回归佐证**：0.20 线（release/0.2）的 CI（run 32817940074，2026-08-25）中，**同一对 musa 用例 `27b_tp4_eager`（inference）与 `35b_a3b_tp4_eager`（serving）均 success**——即 TP4 用例在 0.2 栈可用，迁移到 0.24 栈（torch 2.9.0/torch_musa 2.9.0 组合）后回归，进一步支持问题归属当前栈版本。

---

## 2. 问题①：TP≥2 权重加载挂死

**现象（逐步）**：
1. 以 `tensor_parallel_size>=2` 启动（in-process `LLM` 或 `vllm serve` 均可触发 09-09 形态）；
2. rank0 数秒完成权重加载并继续初始化流程；
3. rank≥1 的显存占用停在 **~293MB**（远低于正常加载量），进程状态 **R（用户态自旋，非 D/锁等待）**，日志无任何推进；
4. rank0 侧随后阻塞在等待 rank≥1 的集体通信上，整个引擎无限挂死，无崩溃栈。

**证据（分两档如实标注）**：
- **[留存证据]** 复现脚本 `/root/tp_test.py`（容器导出，全文见下）：
  ```python
  import os, sys

  def main():
      tp = int(sys.argv[1]); model = sys.argv[2]
      os.environ.setdefault("VLLM_PLUGINS", "fl")
      from vllm import LLM, SamplingParams
      llm = LLM(model=model, tensor_parallel_size=tp, gpu_memory_utilization=0.85, max_model_len=2048, enforce_eager=True)
      out = llm.generate(["用一句话介绍你自己。"], SamplingParams(max_tokens=32, temperature=0))
      print("GEN:", out[0].outputs[0].text, flush=True)
      print("TP%d-OK" % tp, flush=True)

  if __name__ == "__main__":
      main()
  ```
- **[现场观察，dump 未留存]** 09-09 会话中 py-spy 观测 rank≥1 自旋于原生 `aten::copy_ → _copy_from`（H2D 权重拷贝路径）；该次 dump 未进入证据目录，如需原始栈请在复现时重新抓取（`py-spy dump --pid <rank1_pid>`，建议加 `--native`）。
- **[留存证据]** dispatch dump（引擎进程内执行）：`aten::copy_` 在 musa 栈上 `PrivateUse1=False`、注册来源为 PyTorch 原生 `RegisterCompositeExplicitAutograd_0.cpp:3332`——**flag_gems 未注册任何 `copy_` kernel，权重加载全程走原生实现**，即故障点在 torch_musa/驱动的拷贝路径，不在插件可路由范围。
- 注意：证据目录中的 `pyspy_tp0.txt`/`pyspy_tp1.txt` 是 **问题④崩溃后的挂死现场**（见 §5），不是①的现场，首版报告未区分，特此更正。

**精确复现步骤**：
```bash
docker start musa-024-e2e && docker exec -it musa-024-e2e bash
# 对照（预期数秒内输出 GEN: ... 和 TP1-OK）：
python3 /root/tp_test.py 1 /data/models/Qwen/Qwen3-0.6B
# 复现（预期：rank≥1 显存停 ~293MB、≥10 分钟无推进）：
python3 /root/tp_test.py 2 /data/models/Qwen/Qwen3-0.6B
```
09-10 在同一容器复跑此路径时①不再以挂死形态出现，而是漂移为⑤→④（§5/§6）；**建议摩尔在重启宿主机复位驱动后按上述步骤复测**。

**原理（根因假设）**：非纯拷贝 bug——§6 最小复现矩阵证明纯 torch_musa 多进程 H2D 拷贝（含 mccl、flag_gems、mmap 源）全部通过；结合 §6 的"崩溃后驱动状态污染"，假设为 **TP 上下文中某次内核异常污染驱动级设备状态，使后续拷贝在驱动内自旋**。待摩尔在复位后的干净环境复现定位。

**Owner**：摩尔（torch_musa/驱动）。
**潜在解法**：临时绕过 = 只跑 TP1（已采用）；算子层绕过（为 musa 注册替代 `copy_`）理论上可行但见 §7 的否决理由；根修 = 摩尔按本节步骤+§6 矩阵复现定位驱动路径。

---

## 3. 问题②：`vllm serve` 启动失败（fork 冲突）

**现象（逐步）**：
1. 执行 `vllm serve`（任意 TP，含 TP1）；
2. 插件 musa patch 正常加载（日志可见 `Patched ... for MUSA` 系列）；
3. **约 3 秒后**两个 Worker 进程同报 `RuntimeError: Cannot re-initialize MUSA in forked subprocess`，EngineCore 初始化失败，进程退出 `EXIT:1`。

**报错原文（留存证据 `fork_tp2.log`，2026-09-09 12:09:56-59，Worker 侧完整因果链）**：
```
(Worker pid=21233) ERROR 09-09 12:09:59 [multiproc_executor.py:898] WorkerProc failed to start.
(Worker pid=21233) ERROR 09-09 12:09:59 [multiproc_executor.py:898] Traceback (most recent call last):
(Worker pid=21233)   File ".../vllm/v1/executor/multiproc_executor.py", line 865, in worker_main
(Worker pid=21233)     worker = WorkerProc(*args, **kwargs)
(Worker pid=21233)   File ".../vllm/v1/executor/multiproc_executor.py", line 626, in __init__
(Worker pid=21233)     self.worker.init_device()
(Worker pid=21233)   File ".../vllm/v1/worker/worker_base.py", line 331, in init_device
(Worker pid=21233)   File "/root/vllm-plugin-FL/vllm_fl/worker/worker.py", line 432, in init_device
(Worker pid=21233)     current_platform.set_device(self.device)
(Worker pid=21233)   File "/root/vllm-plugin-FL/vllm_fl/platform.py", line 123, in set_device
(Worker pid=21233)     cls.torch_device_fn.set_device(device)
(Worker pid=21233)   File ".../torch_musa/core/device.py", line 68, in set_device
(Worker pid=21233)     torch_musa._MUSAC._musa_setDevice(device)
(Worker pid=21233)   File ".../torch_musa/core/_lazy_init.py", line 107, in _lazy_init
(Worker pid=21233)     raise RuntimeError(
(Worker pid=21233) RuntimeError: Cannot re-initialize MUSA in forked subprocess. To use MUSA with multiprocessing, you must use the 'spawn' start method
(Worker pid=21234) 同上（第二个 worker 同报）
...
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore': 1}
EXIT:1
```
该次启动参数（日志 `api_utils.py:273` 原文）：
```
non-default args: {'max_model_len': 2048, 'tensor_parallel_size': 2, 'gpu_memory_utilization': 0.85,
'disable_log_stats': True, 'enforce_eager': True, 'model': '/data/models/Qwen/Qwen3-0.6B'}
```

**非致命形态（in-process LLM 不死，仅杀上报线程；留存证据 = CI run 34453794059 日志）**：
```
2026-09-10T08:29:04Z (EngineCore pid=1036) Exception in thread Thread-1 (_report_usage_worker):
  File ".../concurrent/futures/process.py", line 246, in _process_worker
  File ".../vllm/utils/platform_utils.py", line 33, in cuda_get_device_properties
  File ".../vllm/utils/platform_utils.py", line 39, in cuda_get_device_properties
RuntimeError: Cannot re-initialize MUSA in forked subprocess. To use MUSA with multiprocessing, you must use the 'spawn' start method
```

**原理**：同一禁令的两个触发面——
1. **worker 进程面**（致命，上面完整栈）：父进程（EngineCore）完成设备探测即已初始化 MUSA，随后 fork 出的 Worker 在 `init_device → torch_musa.set_device → _lazy_init` 处被 `_lazy_init.py:107` 的"fork 后禁止再初始化"断言拒绝；
2. **探测子进程面**（非致命）：vLLM 0.24.0 上游 `vllm/utils/platform_utils.py:33-39` 的 `cuda_get_device_properties` 经 ProcessPoolExecutor（fork 语义）在已初始化 MUSA 的进程里执行，同样触发禁令，杀死 usage 上报线程。
两个事实均为可直接核对的公开代码/异常栈。

**精确复现步骤**：
```bash
docker start musa-024-e2e && docker exec -it musa-024-e2e bash
vllm serve /data/models/Qwen/Qwen3-0.6B \
  --tensor-parallel-size 2 --max-model-len 2048 \
  --gpu-memory-utilization 0.85 --enforce-eager --disable-log-stats
# 预期：~3 秒后输出上文 RuntimeError，进程 EXIT:1（TP1 去掉 --tensor-parallel-size 同样必现）
```

**Owner**：vLLM 上游（fork 探测/worker 启动方式应可配置/可降级）＋ 摩尔（fork 禁令是厂商自我声明约束）。
**潜在解法**：根修（推荐）= 上游对禁 fork 后端退化为进程内直查或强制 spawn（小改动）；临时 = 插件 vendor patch 该函数（有先例形态，未做）；或 torch_musa 放宽 fork 限制。

---

## 4. 问题③：图模式捕获挂死

**现象（逐步）**：
1. `enforce_eager=False`（torch.compile + PIECEWISE cudagraph，capture sizes 至 512）；
2. 权重加载正常（0.5 秒完成），Dynamo 字节码转换正常（3.65 秒）；
3. 打出**最后一条日志**——inductor 模板启发式的 fallback 告警后**完全静默**；
4. 直到 job 60 分钟超时被强制取消，同用例 eager 变体 15 秒出正确结果。

**报错原文（留存证据 = CI run 34453794059，e2e inference job；静默前最后三条日志 + 取消边界）**：
```
2026-09-10T08:30:12Z (EngineCore pid=10233) INFO [default_loader.py:430] Loading weights took 0.50 seconds
2026-09-10T08:30:17Z (EngineCore pid=10233) INFO [backends.py:1089] Using cache directory: .../torch_compile_cache/a4e6d1ea7f/rank_0_0/backbone for vLLM's torch.compile
2026-09-10T08:30:17Z (EngineCore pid=10233) INFO [backends.py:1148] Dynamo bytecode transform time: 3.65 s
2026-09-10T08:30:18Z (EngineCore pid=10233) [rank0]:E0910 16:30:18.43 torch/_inductor/template_heuristics/registry.py:113]
    [0/0] No template heuristic found - template_name=triton::mm, device_type=musa, op_name=mm.
    Available combinations: [('aten::bmm_dtype', 'cuda', None), ..., ('triton::mm', 'cuda', 'mm-ah'), ...].
    Using fallback TemplateConfigHeuristics instance.
<—— 此后 57 分钟 28 秒 零输出 ——>
2026-09-10T09:27:46Z ##[command]/usr/local/bin/docker ...（60 分钟超时，runner 强制取消）
2026-09-10T09:27:47Z Cleaning up orphan processes
```
即挂死点精确定位在 **Dynamo 转换完成后、inductor 编译首个 `triton::mm` 模板（musa 无注册模板、走 fallback）的代码生成阶段**。

**精确复现步骤**：CI 干净 runner 必现——对 PR #457 重跑 e2e（`06b_tp1` 用例含 `enforce_eager=[true,false]` 参数化时）；或手动：
```bash
docker exec -it musa-024-e2e bash
python3 - <<'EOF'
import os; os.environ.setdefault("VLLM_PLUGINS", "fl")
from vllm import LLM, SamplingParams
llm = LLM(model="/data/models/Qwen/Qwen3-0.6B", enforce_eager=False,
          gpu_memory_utilization=0.85, max_model_len=2048)
print(llm.generate(["Hello"], SamplingParams(max_tokens=8)))
EOF
# 预期：打出 template_heuristics fallback 告警后无限静默（对照：enforce_eager=True 数秒出结果）
```

**原理**：慢编译不可能 58 分钟一行不吐——是 musa 上 inductor 编译路径的挂死而非慢。所有摩尔官方 musa 用例均带 `_eager` 后缀，与此一致。
**Owner**：摩尔（torch_musa 编译/捕获路径）。
**潜在解法**：临时 = eager-only（已采用；属 vendor 官方形态，非吞问题）；根修 = 摩尔排查 inductor musa 模板 fallback 后的编译挂死。

---

## 5. 问题④/⑤：IMA 崩溃与 p2p 探测断言（09-10，宿主机状态劣化背景下）

### ④ TP2 dummy 前向 illegal memory access

**现象（逐步）**：
1. 加 `--disable-custom-all-reduce`（绕过⑤）后 TP2 得以推进到 `profile_run`；
2. Worker_TP1 在 inductor 生成代码的 triton kernel **预编译**同步点抛 `InductorError: MUSA illegal memory access`；
3. 此后引擎**不退出**：另一 worker（TP0）永久阻塞在等不回来的 all_reduce 上，EngineCore 每 60 秒刷 `shm_broadcast` 等待告警（观测 ≥6 分钟）；
4. 崩溃后 device 0 泄漏 ~70GB 显存 + 残留 36-40% 幽灵算力，`kill -9` 全部持有进程后**即时**不回收。

**报错原文（留存证据 `tp2_nocar.log`，09-10 20:06:15，Worker_TP1 pid=1390；`tp2_tiny.log` 20:27:20 pid=7101 同签名第二次复现）**：
```
(Worker_TP1 pid=1390) ERROR 09-10 20:06:15 [multiproc_executor.py:1000]
  File ".../torch/_inductor/runtime/compile_tasks.py", line 33, in _reload_python_module
    exec(code, mod.__dict__, mod.__dict__)
  File "/tmp/torchinductor_root/vq/cvqfo5ova4fxjwb54c3qdynrr4dhvdscrutzknzwxy7bbp2ugpyl.py", line 69, in <module>
    triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0 = async_compile.triton('triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0', ...)
  File ".../torch/_inductor/async_compile.py", line 500, in triton
    kernel.precompile(
  File ".../torch/_inductor/runtime/triton_heuristics.py", line 451, in precompile
    self._make_launchers()
  File ".../torch/_inductor/runtime/triton_heuristics.py", line 608, in _make_launchers
    device_interface.synchronize(device_interface.current_device())
  File ".../torch_musa/core/device.py", line 141, in synchronize
    return torch_musa._MUSAC._musa_synchronize()
torch._inductor.exc.InductorError: RuntimeError: MUSA error: an illegal memory access was encountered
MUSA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing MUSA_LAUNCH_BLOCKING=1.
```
崩溃后引擎挂死态（留存证据 `pyspy_tp0.txt`/`pyspy_tp1.txt`，进程 1389/1390 与上同一次运行）：
```
Process 1389: VLLM::Worker_TP0   ← 卡在永远完不成的集体通信
    ... all_reduce (torch/distributed/distributed_c10d.py:2948)
    ... all_reduce (vllm/distributed/device_communicators/cuda_communicator.py:300)
    ... tensor_model_parallel_all_reduce (vllm/distributed/communication_op.py:14)
    ... forward (vllm/model_executor/layers/vocab_parallel_embedding.py:491)   ← dummy 前向里的 embedding all_reduce
Process 1390: VLLM::Worker_TP1   ← 已死于 IMA，回到 zmq 等待循环
    ... dequeue (vllm/distributed/device_communicators/shm_broadcast.py:779) → worker_busy_loop
```
EngineCore 侧每分钟刷屏（`tp2_nocar.log` 20:07:15 起连续 ≥6 条）：
```
(EngineCore pid=1186) INFO 09-10 20:07:15 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds.
This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
```
显存泄漏记录（gmi 实测序列）：`40% | 71228MiB` → `kill -9` 全部持有进程后 → `36% | 71081MiB`（不回收）。
**09-15 复查修正**：8 卡现已全部 `0% | 0MiB`——泄漏在容器停止/进程清退数日后自行回收，首版"仅驱动复位/重启可清"修正为"**kill 后即时不回收；确切回收时点未观测（期间容器被停止、宿主机无负载）**"。另：lowmem 变体（`tp2_lowmem.log`）退出时报告 `718 leaked semaphore objects / 2 leaked shared_memory objects`，一并供参考。

**精确复现步骤**：
```bash
docker exec -it musa-024-e2e bash
vllm serve /data/models/Qwen/Qwen3-0.6B \
  --tensor-parallel-size 2 --max-model-len 2048 \
  --gpu-memory-utilization 0.85 --enforce-eager --disable-log-stats \
  --disable-custom-all-reduce
# 预期：推进到 profile_run 后 Worker_TP1 抛上文 InductorError（第二次复现用 --gpu-memory-utilization 0.08 的 tiny 变体，见 tp2_tiny.log）
```
**Owner**：摩尔（驱动/内核；注意报错虽在 inductor 同步点暴露，但 IMA 是设备侧异常，且独立于编译参数两次复现）。
**潜在解法**：临时 = 不跑 TP2（已采用）；根修 = 摩尔以 `MUSA_LAUNCH_BLOCKING=1` 复跑定位实际越界内核。

### ⑤ p2p 探测断言

**现象**：不加 `--disable-custom-all-reduce` 时，TP2 init 在 custom all-reduce 的 p2p 探测处直接断言退出（早于权重加载）。

**报错原文（留存证据 `tp2_forensic2.log`，09-10 20:00:05，Worker pid=457）**：
```
(Worker pid=457) ERROR 09-10 20:00:05 [multiproc_executor.py:898]
    if not current_platform.is_rocm() and not _can_p2p(rank, world_size):
  File ".../vllm/distributed/device_communicators/custom_all_reduce.py", line 39, in _can_p2p
    return torch.cuda.can_device_access_peer(
  File ".../torch/cuda/__init__.py", line 627, in can_device_access_peer
    raise AssertionError("Invalid device id")
AssertionError: Invalid device id
[rank0]:[W910 20:00:06 ProcessGroupMCCL.cpp:1095] Warning: destroy_process_group() was not called before program exit, which can leak resources.
(EngineCore pid=254) ERROR 09-10 20:00:07 [core.py:1231] EngineCore failed to start.
```
**原理**：状态依赖（09-09 同环境可通过此探测）；与摩尔官方用例一律 `disable_custom_all_reduce: true` 相互印证——vendor 已知该路径不可靠。
**复现**：④步骤去掉 `--disable-custom-all-reduce`。
**Owner**：摩尔。
**潜在解法**：临时 = `disable_custom_all_reduce=true`（vendor 约定，已采用）；根修 = 摩尔修正 musa 后端下 `can_device_access_peer` 的 device id 校验。

---

## 6. 最小复现矩阵（关键阴性结果）与状态漂移

无 vllm 的多进程复现矩阵（脚本：宿主机 `/root/scripts_debug/repro_musa_copy_hang.py`（A–D）、`repro_musa_copy_hang_v2.py`（E–F），随摩尔工单附件交付，不进仓库）。

**结果原文（留存证据 `repro_result.log` / `repro_v2_result.log`）**：
```
==A-D==
torch_musa devices: 8
[A_pure_multiproc] PASS (7s)      # 双进程 spawn，各自 pinned + non_blocking H2D 到自己的卡
[B_mccl_own_dev] PASS (66s)       # A + mccl init（world=2）
[C_mccl_all_dev0] PASS (9s)       # B 但双 rank 同卡（对照）
[D_mccl_nopin_block] PASS (8s)    # mccl + 无 pin + 阻塞拷贝
DONE:0
==E-F==
torch_musa devices: 8
[E_flaggems_mccl] PASS (12s)      # B + import flag_gems（kernel 注册 + backend 激活）
[F_mmap_mccl] PASS (13s)          # mccl + safetensors mmap 源、256 个 4MB 张量循环拷贝（模拟权重加载）
DONE:0
```

**全部通过** → 纯 torch_musa 多进程拷贝本身不挂，故障需要完整 vLLM TP 上下文。

**状态漂移记录**：宿主机横幅自 09-09 起提示 `*** System restart required ***`（至今未重启，`uptime -s` = 2026-06-27 22:54:36）。09-09 TP≥2 表现为①挂死；09-10 同容器同代码漂移为⑤断言→④IMA，且④的泄漏累积后使同宿主机的 CI unit 阶段从健康基线 77 秒劣化到 1.5 小时+（run 34477790128 对照记录）。**上报前建议：重启宿主机、复位驱动，在干净状态按 §2/§5 复现步骤 + A–F 矩阵顺序复测**。

## 7. 为什么"算子黑名单/注册替代算子"对本报告的问题不适用

本仓库存在 sglang 式算子黑名单机制且被广泛使用（`dispatch/config/*.yaml` 的 `flagos_blacklist`/`oot_blacklist`，env `VLLM_FL_FLAGOS_BLACKLIST_APPEND`；nvidia.yaml 甚至拉黑过 `copy_`/`to_copy`）。但：

- dispatch 系统只管理 5 类算子（attention/rms_norm/rotary/silu_and_mul/topk_softmax），**权重加载的 `copy_` 不在其中**，且此栈 flag_gems 根本没注册它（§2 dispatch dump）——没有可拉黑的对象；②是进程模型问题、③是编译捕获问题、④⑤在内核/驱动层——均不可通过算子路由绕过。
- 理论方案"为 musa 注册替代 `copy_` kernel"（`torch.library` PrivateUse1）需覆盖 dtype cast/broadcast/non-contiguous/async pinned H2D 全部语义，且若根因是驱动状态污染（§6 假设）则注册什么都会被污染。**不建议在摩尔定性前做**。

## 8. CI 现状

CI（PR #457，09-14 run 34815101547）：**全绿**——Setup / unit / functional / e2e（`qwen3/06b_tp1`，eager）/ build / lint / CodeQL；benchmark 配置为空、空跑通过。该绿色代表"收缩后测试面无问题"，五个故障仍然开放（见 §0 恢复条件）。

## 9. 附件索引（2026-09-15 修订：全部已导出宿主机）

宿主机 `/root/fl-musa-evidence/`（同时保留在容器 `musa-024-e2e:/root/fl-musa-evidence/`）：

| 文件 | 内容 |
|---|---|
| `fork_tp2.log` | ②完整报错栈（09-09 12:09，worker fork 冲突，含启动参数） |
| `tp2_forensic.log` / `tp2_forensic2.log` | 09-10 19:53/20:00 取证运行；forensic2 含⑤ p2p 断言完整栈 |
| `tp2_nocar.log` | ④第一次 IMA 完整栈 + 崩溃后 shm_broadcast 挂死记录（20:06） |
| `tp2_tiny.log` | ④第二次 IMA 同签名复现（20:27，0.08 显存变体） |
| `tp2_lowmem.log` | lowmem 变体（退出时 718 个泄漏信号量） |
| `pyspy_tp0.txt` / `pyspy_tp1.txt` | ④崩溃后挂死现场（TP0 卡 all_reduce / TP1 回 zmq 等待；native dump） |
| `tp1_inf.log` / `tp1_srv.log` | 09-09 TP1 健康记录（in-process / serve） |
| `tp1_today.log` / `tp1_lowmem.log` | 09-10 TP1 对照记录 |
| `repro_result.log` / `repro_v2_result.log` | A–F 矩阵结果原文 |

复现脚本：`/root/scripts_debug/repro_musa_copy_hang.py`、`repro_musa_copy_hang_v2.py`、`/root/tp_test.py`（§2 全文引用）。
CI：run 34453794059（③静默实录 + ②非致命形态，日志已另存）；run 34477790128（unit 劣化对照）；run 32817940074（0.20 线 TP4 用例通过的回归对照）。
