# AlphaRush 模型部署

模型环境位于 WSL `/home/<user>/alpharush/.venv`，与 <other-project> 的环境相互独立。历史模型推理阶段将现有权重作为只读输入复用，没有下载新底座、加载 other-project adapter 或执行参数更新。8B 的最多一步新 LoRA 检查另有预注册、参考和回执，实际结果见 `REPORT.md`。您后续已授权先用 24B 开始第一关奖励训练，新阶段见 `LEVEL1-TRAINING.md` 与独立 `runtime/rl/level1-24b-phase1/`；旧双模型推理速度对照保留为历史证据。

| 模型 | 指令底座 | 设备 | 推理格式 |
|---|---|---|---|
| 8B | Ministral-3-8B-Instruct-2512-BF16 | RTX 5080，UUID `GPU-750ffce3-8683-57c4-8737-35bd45ad462f` | NF4，BF16 compute |
| 24B | Mistral-Small-3.2-24B-Instruct-2506 | RTX 5090，UUID `GPU-4d9f95ae-dfba-0ba5-9aad-09372fa19208` | NF4，BF16 compute |

两底座的原始架构都是 Mistral3 图文包装器。worker 正确加载该包装器，然后只提供文字 token；没有把图文权重误当作未经映射的独立语言模型。

请通过 `tools/run-model.ps1` 的项目门禁启动模型作业。`tools/model-worker.py` 是供启动器调用的底层工具：默认读取一份请求、完成后退出；`--requests` 接受最多 6 份请求并共享一次模型加载。长期 HTTP 服务必须显式指定 `--serve`。

请求格式是 `{id, system, user, labels}`。`user` 使用游戏的精确状态与完整合法菜单。返回包含全部标签的 `p`、`logp`、合法 `choice`、原始 user 字节 SHA256、模型 revision、GPU UUID、耗时和峰值显存。每个标签按完整输出序列加 EOS 的条件 log likelihood 评分，再仅对合法菜单归一化；AA 等标签不会被截成第一个 token。超长提示会拒绝，不能静默截断。

首关 seed 1001、tick 361 的同一真实局面已有两模型部署验证。请求为 3555 token、26 个合法选项；各做 2 次 warmup 和 3 次计时。完整原始文件位于 `runtime/rl-models/native-matched-batch5.json`、`native-8b-result5.json`、`native-24b-result5.json`。两模型都选择 B（在 holder 15 建箭塔），各自 5 次分布完全一致。

| 部署验证 | 8B / 5080 | 24B / 5090 |
|---|---:|---:|
| 冷载权重 | 19.979 秒 | 54.988 秒 |
| 三次完整分布计时中位数 | 1.874 秒 | 1.630 秒 |
| 峰值模型 allocated 显存 | 6851 MiB | 15229 MiB |

这里计时包含文字 tokenization、一次共享提示计算和全部合法选项评分，不是每 token 生成速度。模型大小与显卡同时不同，这组数字描述这两套实际部署配置。一个局面的相同选择不能判断哪个模型更会通关；24B 的选择概率更集中也不是质量提升证据。关卡续跑的真实结局由项目比较计划另行验证。

完整底座 SHA256 核验由 `tools/create-model-manifests.py` 单次只读扫描，结果缓存到 `/home/<user>/alpharush/manifests`，并复制至 `runtime/rl-models/model-8b-manifest.json` 与 `model-24b-manifest.json`。包含全部权重分片、索引、配置、tokenizer、chat template 和下载 revision 元数据。扫描安排在 GPU 计时结束后；原始 5 次计时文件保持原样，`native-manifest-binding.json` 明确记录事后身份绑定。以后 worker 启动按文件大小与修改时间验证此完整审计，每响应记录 manifest SHA256，避免重复读取数十 GB 权重。

训练态对齐所需的 8B 基线另有 4 个原生 reference：首关 fork、训练 A1/A2、第二关无奖励的初始状态。它们是 3555、4285、3920、3908 token；事前独立计划允许 6144，超限拒绝截断，不能混入上面的部署时延比较。该一次推理限制为 600 秒、4 个 CPU 线程，使用 Linux GPU 锁和 STOP 监控，实际墙钟 30.17 秒、参数更新 0 次。原始完整分布为 `native-reference-8b-baseline.json`；`native-reference-8b-baseline-enriched.json` 保留原 p/logp，并附 CPU 重新编码核验得到的完整 prompt-token-ids SHA256 与真实 tokenizer 文件 SHA256。`native-reference-identity-binding.json` 标明这些身份字段是在 GPU 推理后补充的；它们不能视为模型训练或训练收益证据。

初次冒烟发现缓存 crop 接口的正数兼容语义，严格长度检查拒绝了无效分布；已改为负数删除 suffix，错误日志保存在 `/home/<user>/alpharush/native-*-smoke-*.log`，正式 5 次结果来自修复后的代码。
