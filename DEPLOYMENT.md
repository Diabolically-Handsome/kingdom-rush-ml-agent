# Kingdom Rush 1 本地部署

本部署把 Lumi_Nox 的游戏内 TCP 桥接接口接入您安装的第一代 Kingdom Rush，并建立独立的 AlphaRush RL 框架。5080 的 8B 与 5090 的 24B 已完成实际推理及同一首关局面的续打对照；原生动作、重放和小型 CPU 策略训练已验证。8B 的独立 LoRA 检查结果见 `REPORT.md`，正式长训没有开始。下方普通游戏入口默认使用本地规则策略，不需要模型 API 密钥。

您已进一步授权先用 24B 进行第一关奖励训练。独立有界首轮见 `docs/LEVEL1-TRAINING.md`，实际结果在 `REPORT.md` 与 `runtime/rl/level1-24b-phase1/latest-training.json`；奖励为通关 `1 + 剩余生命 / 20`、失败 `-1`，工程错误无奖励。双击 `RL状态.cmd` 可查看实际更新和验收状态。

## 部署位置

| 项目 | 路径 |
| --- | --- |
| 工作目录 | `C:\Users\<user>\Documents\AlphaRush` |
| 代码仓库 | `C:\Users\<user>\Documents\AlphaRush\Lumi_Nox` |
| 独立 Python 环境 | `C:\Users\<user>\Documents\AlphaRush\.venv` |
| 游戏程序 | `D:\SteamLibrary\steamapps\common\Kingdom Rush\Kingdom Rush.exe` |
| 原始程序备份 | `D:\SteamLibrary\steamapps\common\Kingdom Rush\Kingdom Rush.exe.bak` |
| 本机控制接口 | `127.0.0.1:9878` |

上游项目：[MIO-456/Lumi_Nox](https://github.com/MIO-456/Lumi_Nox)。对应说明：[王国保卫战逆向工程笔记](https://github.com/MIO-456/Lumi_Nox/blob/main/docs/games/kingdom-rush-reverse-engineering.md)。

## 从哪里开始

先双击本目录的 **启动游戏.cmd**。游戏窗口会正常显示，启动助手隐藏运行，桥接服务在游戏进程内运行。

之后，您可以按目的选择一个入口：

| 入口 | 用途 |
| --- | --- |
| **交互控制.cmd** | 确保游戏启动后打开命令行控制客户端；输入 `ping` 测试响应，输入 `state` 查看状态，输入 `quit` 退出客户端。 |
| **观察状态.cmd** | 查看游戏推送的金币、生命、波次和单位数量；进入关卡后才有完整战斗状态。按 `Ctrl+C` 停止观察。 |
| **AI当前关卡.cmd** | 您先在游戏内手动进入要测试的关卡，再双击此入口。它先检查界面，再由规则策略控制当前关卡。按 `Ctrl+C` 停止控制，游戏继续运行。 |
| **还原游戏.cmd** | 您先正常退出游戏，再双击此入口，把原始 EXE 备份复制回游戏程序，并保留备份。 |

AI 入口会执行真实建塔、升级、施法等操作，当前关卡的胜负可能由游戏正常写入存档。它只处理您当前进入的关卡，不会自动选关、循环刷关或启动训练。对战记录和策略日志由上游代码写在 `Lumi_Nox\games\kingdom_rush\logs`，这类记录不代表模型参数训练。

交互入口里 `state`、`ping` 用于读取和检查。`build`、`upgrade`、`sell`、`wave`、`hero`、`power` 则会实际操作游戏；客户端启动后会显示参数说明。建议先从 `ping` 和 `state` 开始。通常一次使用一个控制客户端即可。

## 可选模型 API

没有 `deployment.env` 且您没有在当前 Windows 环境显式设置 `KR_NO_LLM` 时，启动器默认设置 `KR_NO_LLM=1`，使用本地规则策略。本地 `deployment.env` 的配置优先于继承的环境变量。

如果您之后希望接入模型，把 `deployment.env.example` 复制为同目录的 `deployment.env`，填写以下配置即可：

```dotenv
KR_GAME_EXE=D:\SteamLibrary\steamapps\common\Kingdom Rush\Kingdom Rush.exe
KR_NO_LLM=0
KR_ARK_API_KEY=填写您的实际密钥
KR_MODEL=填写您账号可用的模型标识
KR_API_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions
```

默认接口为上游使用的火山方舟 Chat Completions 接口。模型必须由您所配置的服务提供，支持该策略代码使用的工具调用格式；仅更换 URL 和模型名不保证服务兼容。启用后会向所配置的模型服务发送策略所需的游戏状态与对战上下文，并可能产生 API 费用。默认规则策略不需要这项配置。

配置只在此次启动和子进程内生效，不修改 Windows 全局环境。它采用 `KEY=VALUE`，允许整行注释和成对引号，不执行脚本。启动器不会打印密钥。请将实际 `deployment.env` 留在本机，不放入公开仓库、截图或分享材料。恢复规则策略时把 `KR_NO_LLM` 改回 `1`。

## 游戏更新与还原

补丁会改写游戏 EXE，在原始备份基础上注入 Lua 桥接服务。**请保留 `.exe.bak`。**

“[用户原话已省略 / user's message omitted]”会检查游戏已退出、文件夹只有目标 EXE、原始备份存在且唯一，再调用上游 `patch --unpatch` 并核对还原文件与备份的 SHA256。检查失败时会停止并说明原因，不结束游戏进程，也不删除备份。

Steam 更新或“验证游戏文件完整性”可能恢复原始 EXE，移除桥接补丁；游戏仍可运行，但控制客户端无法连接。卸载游戏也会移除游戏目录内的补丁，通常也包括备份。还原后如果您还需要控制接口，须重新注入补丁；游戏版本更新后先核对兼容性，不应把旧版本备份直接套到新版游戏上。

## 常见情况

- **客户端等待连接或超时**：确认游戏已启动、没有加载报错，且当前 EXE 仍包含桥接补丁；接口默认是本机 `127.0.0.1:9878`。
- **AI 提示先进入关卡**：在游戏中手动选关，等关卡画面出现，再重新运行 AI 入口。
- **找不到游戏程序**：在 `deployment.env` 中设置正确的完整 `KR_GAME_EXE` 路径。
- **提示未配置 API 密钥**：保持 `KR_NO_LLM=1` 即可使用规则策略；只有显式启用 API 策略时才需密钥。
- **还原入口提示游戏仍运行**：正常退出游戏后重试，不要在游戏进程运行时覆盖 EXE。

## 最初桥接部署的验证记录

以下表格保留最初桥接部署时的检查范围；后续 AlphaRush 原生关卡、模型与训练验证见下方 RL 部分和 `REPORT.md`。最初检查时尚未加载关卡。

| 检查项 | 结果与证据 |
| --- | --- |
| 仓库版本与本地修改 | 使用上游仓库；本地补充 Steam 游戏路径、无密钥规则策略和可选 API 配置支持。准确版本与文件证据见 `runtime\deployment-manifest.json`。 |
| 独立环境及依赖 | Python 3.12.10；独立 `.venv` 安装 requests，依赖冻结在 `requirements-local.txt`；`pip check`、Python 代码编译检查、无密钥命令行帮助通过。 |
| 原始 EXE 备份与补丁 | 备份与原始 EXE 的 SHA256 相同；注入后原始 1476 个资源逐条 SHA256 与引擎前缀保留，原始和补丁文件 CRC 检查通过。 |
| 游戏启动与 TCP 连接 | 已实际启动游戏并收到 Bridge v5.13 欢迎消息，连接 `127.0.0.1:9878` 成功。 |
| `ping` / 当前界面 / 状态读取 | `ping` 通过；`detect_screen` 返回 `screen_info`、当时界面为 `unknown_nil`；`get_state` 返回 `game_state` 协议，但带有 `game.store not found` 错误。尚不能据此确认关卡内状态字段正确。 |
| 启动器检查 | PowerShell 语法与 AI 界面检查代码语法通过；游戏、交互、观察、AI 四种入口的只检查配置模式通过。实际 AI 入口在未进入关卡时按预期拒绝控制，返回退出码 2。还原入口在游戏运行时按预期返回非零，游戏退出后配置检查通过；没有实际执行还原。 |
| 游戏控制动作验证范围 | 尚未验证建塔、升级、技能、英雄移动或实际战斗。 |
| AI 实战 / 通关 / 强化学习训练 | 本次没有运行 AI、自动刷关或训练，未验证通关表现。 |

本次文件校验值：

```text
原始 EXE / .exe.bak SHA256
ebb4a3a9fd4fa6a30fb98e2b39fb32f0eff8630d47a3951b658339c26ce91e7b

注入补丁后的 EXE SHA256
41aa2bdc8e42e3b79ac96cfefeff12a6adf2c18b4f490a78228b6afec92797b8
```


## AlphaRush RL（本次新增）

- 双击 `RL状态.cmd` 查看本次环境、训练和双模型对比状态。
- `验证游戏环境.cmd` 在隔离游戏和独立存档里验证首关基础动作与分支重放。每次证据保留独立目录；新证据后需要明确重新冻结，旧计划不会自动通过。
- `小规模训练验证.cmd` / `模仿训练验证.cmd` 分别运行真实数据上的小型 CPU RL / SFT 验证；它们不会训练 8B/24B，也不会启动正式长训。
- `对比模型.cmd` 是有预算的实际双模型流程：5080→8B，5090→24B；同状态、完整合法选项、两次预热和三次计时，并在游戏里续打、核真实回执及重放。部署验收消耗本次预定的两个 GPU 作业；后续新比较需要新的计划/预算，代码不会自动扩容或删除历史。
- 8B 的真实 LoRA 训练模式另用独立、最多一步的流程核验；具体结果与限制追加在 `REPORT.md`。不能用小型 CPU 更新或纯推理成功代替这一结论。

当前可靠范围是第一关 Normal 的等待、四种基础塔、发波及固定种子冷重放。英雄、技能、升级、出售、多局并行和全关卡表现仍待验证。第二关仅取无奖励初始状态作为验证锚点；第三关和留出种子继续保留。大规模正式训练没有开始。

程序、模型底座和旧项目环境各自保持独立；每个模型由完整 SHA 清单确认，模型进程按作业加载、完成后退出。所有游戏测试使用 `runtime/rl-engine` 里的副本；原 Steam 游戏与玩家存档不作为训练环境。
