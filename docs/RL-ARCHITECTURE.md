# AlphaRush RL 部署架构

这份部署把 <other-project> 的规则脑、字母选项脑、真实执行回执和受约束分支训练拆成独立的 KR1 工程。交接文档的另一个项目成绩、旧授权、奖励和学习率都不构成 KR1 的证据或授权。实战验证针对第一关，另读第二关的初始状态作为无奖励验证状态；第二关没有做续打验收。您已明确 8B/5080 与 24B/5090 两种方案，并在完成部署后授权先用 24B 开始第一关奖励训练；新增独立阶段及奖励见 `LEVEL1-TRAINING.md`。旧“正式训练稍后”只描述先前部署阶段，不能阻止本次新授权范围。

## 数据和控制流程

```
KR1 原生状态/分步桥接 → 环境层 → 规则脑产生合法选项
                                     ↓
                         选项脑：提示 + 菜单 → 一个字母
                                     ↓
                        执行器 → 原生状态核验回执
                                     ↓
             决策 journal / 原生结局 / 重放证明 / 分支样本
                                     ↓
                  SFT → DAgger → 受约束 pi_adv RL
```

规则脑负责常规推进、合法性、塔位与价格、具体坐标等低层执行。选项脑决定关键时刻选哪项：建塔、升级、技能、英雄位置、攒钱等。模型不能收到未来结局或内部 RNG；重放工具可单独保存 RNG 和内部状态。

每次决定记录 `decision_source`（rules/model）、提示与菜单 SHA、完整合法字母、最终字母、实际动作、原生回执、帧数和结局。报告分别统计规则决定数、模型决定数与各自占比；脚本变强、菜单改动和模型更新分别报告。模型尚未选择时，不能把脚本成绩写成模型学会。

## 真实环境验收

`runtime/rl/native-evidence.json` 的 `schema_version` 为 1，`gates` 中包含以下四项，每项必须有 `status: verified`、工作区内的 `artifact_path` 和对应 `artifact_sha256`，真实训练才可发车：

1. `exact_state`：生命、金币、波次、敌人、塔、英雄与技能状态能读到，并核实字段语义。没有读到的状态标 pending，不能由名字猜值。
2. `action_receipts`：合法动作和拒绝动作都有原生前后状态回执。发出请求不等于操作完成；建塔、升级、卖塔、集结点、增援、陨石、英雄操作分别核验。
3. `deterministic_replay`：同一个 seed 和动作 journal 重跑，原生状态、动作顺序、帧时序和终局一致。只启动两次或只比最终生命不够。
4. `branch_replay`：同一关键状态可重建，重建前缀、提示字节和菜单完全一致，改选候选后可继续到真实原生终局。每个候选至少一个有效分支；工程错误不能作为奖励 0。

暂停、精确分步、随机种子、隔离重置、无头/加速/并行在实测前均视为 pending。只有桥接端点存在或 ping 成功不能证明其运行语义。实时多帧推进中技能动作必须定义应用帧和容许误差；如果时间或随机重放无法一致，分支 RL 保持关闭。

已有 HTTP 控制接口、基础启动部署和本地桥接是起点。API 的完整性、游戏内部运行和原版一致性由本项目实测记录确认，部署文档不预先宣布通过。游戏交互、优化器真实参数更新、大模型 LoRA 更新、能力改善是四种不同验收。

## 精确 pi_adv 目标

这沿用交接项目实际源码的有限合法动作期望目标；它不是 sampled PPO 或带比值裁剪的 RLOO。

对于同一岔路的合法字母集合 `L` 和候选 `C`，各候选都有真实结局价格 `R(c)`。定义：

```
Rbar = mean(R(c), c in C)
Rt(a) = R(a)                  a in C
Rt(a) = min(R(c), c in C)     a in L but not C
adv(a) = Rt(a) - Rbar
J(theta) = sum(pi_theta(a) * adv(a), a in L)
```

`pi_theta` 是全部合法字母 logits 上的 softmax。候选奖励全相等时该组不贡献策略梯度；仍受 KL 约束。最小化：

```
loss = -mean(J_fork)
       + beta_fork * mean(KL(pi_theta || pi0) on forks)
       + beta_anchor * mean(KL(pi_theta || pi0) on A1 + A2)
```

fork β 只在 KL 超目标时升高；anchor β 可双向调节。A1 是 SFT 锚点，A2 是冻结 π0 的普通训练决定；二者都只能来自训练池，分开报告 KL 和 argmax 改变率。KL/熵/学习率硬阈值由本次独立配置和真实优化标定确定，不能把另一个项目的 `3e-7` 宣称为 KR 最佳值。

训练前核 π0 在训练加载路径和推理 broker 路径的完整合法分布、字母 token 映射、提示 SHA 一致。训练中核熵下限、分叉和锚点 KL、非有限值；有工程错误先停。保存候选不表示模型已经变好。小规模 CPU NumPy 更新只能验证这一数学目标和真实数据链；它不能证明已选择的大模型或 GPU 训练已就绪。

奖励以冻结 scorer 契约为准。建议围绕是否过关、剩余生命和星数制定明确目标；不得把缩短失败时间隐含奖励成进步。raw outcome、price、scorer SHA、prompt SHA 和分支来源随样本保存；缺分支或回执不一致的组拒绝训练。

## 数据池与留出纪律

`configs/pools.json` 固定：训练 seed 1001–1010 / level 1；validation seed 2001–2004 / level 2；heldout seed 3001–3004 / level 3；never_train seed 4001–4004。未列出的种子和关卡默认永不训练。本次只实战验证第一关；第二关 seed 2001 的初始状态用于无奖励监测，没有读取续打结局；heldout 未消费。

训练组和 A1/A2 锚点都按 seed 和 level 双重核查。validation 只用于监测，不能参与梯度；heldout 留给冻结候选的一次判定，任何训练、重标注或 anchor 构建不能读取其结局。`claim_heldout_evaluation` 在读结局之前追加一次 reservation；中止后也不自动重开。新增备用留出需正式记录新协议。

## 独立启动与账本

运维层 `alpharush_rl/ops.py` 不导入模型/GPU库。`preflight` 只读，检查代码和配置 SHA、数据池、STOP、锁、预算、真实环境证据。配置默认没有生成 pins；只有显式 `freeze` 后核 SHA 通过才叫冻结。训练器、环境、scorer 及其价契约应由最终 CLI 作为 extra_paths 一并 pin，改代码需重新核证据、正式重新 freeze 并留历史。

所有状态位于 `runtime/rl/`，与 <other-project> 独立：

- `STOP` / `ENGINEERING-STOP`：拒绝新发车并停止运行中的作业。代码不会自动删除。
- `job.lock`：独占进程锁；异常遗留锁需要核实，不能静默删掉。
- `ledger.jsonl`：只追加 open/close、代码身份、已消耗墙钟、错误；未闭合账本拒绝新发车。
- `runs/<id>/`：独立输出、事件和进程回执。`receipt.verified=false` 表明这只是进程/预算回执，科学结论另验。
- `pins-history.jsonl` / `heldout-audit.jsonl`：记录 freeze 和一次留出消费。

`launch(config_path, callback, ...)` 把模块级可序列化 callback 放入独立 CPU 子进程，检查 STOP，超过墙钟就终止该作业；Windows 入口必须有 `if __name__ == '__main__'`。`job_context` 是合作式替代方案，优化循环每步必须 `context.check()`；它不能对一个不配合的阻塞函数承诺硬墙钟。callback 返回简短摘要，大样本写到独立输出目录。

本次 CPU smoke 每作业 45 秒，累计按种类分别有界。`cpu-smoke` 只验合成优化器；`cpu-rl-smoke` 必须四项原生证据和被审计的真实训练样本。训练文件 SHA 必须与原生证据对应，文件内 pool_registry 必须与外部冻结池相同，validation 状态另核身份且不能有奖励字段。合成数据不能替代原生证据，也不能进入真实游戏效果统计。通用 GPU 作业、自动长期 serving 和正式训练仍为 disabled；您授权的双模型有界推理和 8B 最多一步 LoRA 检查使用独立入口，结果与失败均记入 `REPORT.md`。这些流程不借用 <other-project> 的旧 GO、30 小时预算或在运行作业。

## 两种模型部署的比较

`configs/models-comparison.json` 保存预先制定的比较计划，最终代码/config SHA freeze 也覆盖该文件。两边均用本机底座重新加载 4bit NF4，adapter 为 null；8B 为 `Ministral-3-8B-Instruct-2512-BF16`，固定 RTX 5080 UUID；24B 为 `Mistral-Small-3.2-24B-Instruct-2506`，固定 RTX 5090 UUID。配置状态 `verified_scoped_inference` 仅指本次相同首关提示的有界推理通过；LoRA 更新、长期服务和正式训练另验。

速度对比用相同原生状态、精确提示字节、完整合法菜单和顺序、context 内容与上限、seed/关卡/难度/帧窗口。相同文字经不同 tokenizer 的 token 数可能不同，均单列记录；超出上限双边拒绝，不静默截断一边。每模型两次 warmup 独立报告，再报告每状态三次测量的中位数/p90、完整请求时间和最终整局墙钟。模型加载与排队耗时也单列。

两模型家族、版本、tokenizer 和显卡都不同，结论是两种部署方案的比较，不能把差异因果归为参数数目。合法字母掩码产生的 0 非法选择只是接口约束；未进行非受限生成校准时，原始非法率写“未测”，不能宣称模型 0 非法。

只有两边 broker 都通过、并在完全相同规则骨架下只换选项模型，全程实际操作并核原生回执/结局，才能比较自主决定占比、胜负和剩余生命。离线提示选择和三次速度调用不能代替整局结果。初始比较只是训练池第一关一个决定点的探索，之后仍是冻结规则续打；模型自主整局没有证明。未来开发对比使用 validation 池（seed 2001–2004、关卡2），须先验该关动作/重放，状态为 pending；两边都不得参与梯度。heldout 关卡3及其 seed 保留给未来一次正式判定。

`对比模型.cmd` 创建新的配对计划，按序调用 8B 和 24B 的有界单批推理，再交 CLI 做原生选择续打和冷重放。单臂入口为 `tools/run-model.ps1 -Model 8b|24b -Directory <已准备目录>`；`-CheckOnly` 仅读验证与 WSL busy 查询，不生成作业。每批最多6条、最多600秒，本次整个比较最多2个作业/1200秒；超出本次预算需记录新协议，不能悄悄重跑。

有界推理逐次核本项目 STOP、代码/config pins、原生证据、当前 owner_words、相同 plan/messages/menu、物理 GPU UUID 和繁忙条件，独占 Windows/WSL 本项目锁。Linux supervisor 监控 STOP 和墙钟，外层 `timeout` 再加一层；停止只终止它自己的 worker 进程组。账本 `model-inference-ledger.jsonl` 和每作业 `receipt.json` 保存正常/失败记录，未闭合账本拒绝发车。辅助 WSL 进程隐藏窗口；不会默认 `--serve` 或启动训练。

完成框架后按序验证：原生环境→纯脚本基线→决策来源统计→SFT/DAgger数据链→小规模真实分叉与参数更新→真实模型标定→您安排正式训练→一次留出判定。各阶段的证据追加到根目录 REPORT.md。
