# Dialogue QA Benchmark

这个仓库从对话及其中提供的代码、工具、失败和测试记录中，构建有证据依据的 QA。QA 构建不连接记忆服务，也不读取仓库补充历史事实。另有可选的任务试验，将已审核的 QA 延伸为新的仓库需求。

## 方法

流程把对话看成按时间排列的证据流：

1. 规范化可见消息、工具记录、代码观察、补丁、失败和测试结果。
2. 在内存中回放提供的补丁，建立轻量证据图。节点表示对话/代码对象和历史版本，边表示明确引用、修改、失败、反馈、测试和先后关系。
3. 从整张图中选择小的相关子图。相隔很远的记录只在明确指向同一个文件、函数、决定、失败或需求时组合。
4. 从小块证据中抽取带来源的事实，再只根据选中的证据生成和审核 QA。
5. 分开保存候选、审核和发布结果，方便查看被拒绝或尚未完成的题目。

这里的图只复原输入对话暴露出的信息，不等同于当前代码仓库。高价值问题应依赖过去的修改、反馈、失败、测试、决定或跨阶段对话，而不是当前仓库可以直接读出的表面事实。

普通和代码 QA 共用六类历史用途，保留独立配额。普通题侧重已记录的决定和观察；代码题将历史联系到实现或测试行为。

| 类型 | 保存的历史知识 | 派生开发需求 |
|---|---|---|
| `constraint_followthrough` | 已确认约束及范围 | 新增功能或重构时继续遵守 |
| `correction_update` | 旧规则及后来的局部纠正 | 在适用范围使用修订后的规则 |
| `external_state_application` | 用户侧或环境中的实际事实 | 根据已观察条件增加适配 |
| `failure_avoidance` | 失败过的方案及触发条件 | 实现新行为时避免重现失败 |
| `verification_reuse` | 实际测试或实验结论 | 据此选择实现边界和回归检查 |
| `compatibility_preservation` | 旧调用者需要保留的行为 | 增加新能力并维持必要兼容 |

每次请求静态指定一种用途，选题和 QA 生成只接收该用途的定义，单独的简短审核检查目标是否一致。证据、原子性和完整性仍分别检查。需求生成沿用同一用途，必须将实际注入的历史答案联系到新的实现选择和可观察验收行为。旧 QA 类型不再接收；旧运行需要重新生成 QA 后再派生需求。

难度由本地静态规则计算：必要阶段/版本数量、图上的关系距离，以及是否跨越失败→修改→验证链。模型不负责决定难度。

## 安装和运行

[真实任务构造示例](examples/task-construction-review.md) 包含实际 QA、需求草稿、失败过程和构造成本，便于对照代码审阅。

需要 Python 3.9 及以上：

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests -v
```

静态检查（不联网）：

```sh
python -m dialogue_benchmark.cli examples/dialogue.json \
  --output runs/example --qa-mode both
```

使用大模型生成时，需要显式提供 HTTPS endpoint 和模型，并确认允许发送选中的证据：

```sh
export BENCHMARK_API_KEY='...'
python -m dialogue_benchmark.cli examples/dialogue.json \
  --output runs/llm-example \
  --qa-mode both \
  --allow-network \
  --endpoint 'https://example.invalid/v1/chat/completions' \
  --model 'your-model'
```

输入可以是统一 JSON 或支持的原始 rollout。记录可以包含可见消息、成对的工具调用/结果、代码观察和成功补丁。输入中的路径只作为证据处理，程序不会按路径读取磁盘文件。

支持 OpenHands 公开的 `session.jsonl`，保留成对工具记录和已成功的文件编辑；也支持仅含 user/assistant 消息的 `dialogue.json` 列表，后者本身不含工具历史。工具正文保留真实换行，完整编辑内容进入版本证据，不在工具元数据中重复呈现。

每次运行写入独立目录，包含规范化证据、范围、事实、候选题、审核信息和公开 QA 视图。运行目录只供本地使用并被 Git 忽略。可以打开 `viewer/index.html`，或先运行 `viewer/build_data.py` 查看保存的运行结果。

## 仓库任务试验

先运行 `python convert_session.py /path/to/session.jsonl --output runs/episode/input`，再将生成的 `dialogue.json` 交给 QA 流程。转换复用已有解析器，保留 OpenHands 工具结果、文件变更和原始记录位置；`conversion.json` 保存输入哈希与数量。抽取代码历史题时优先使用原生 session 日志，而非只有可见消息的导出。

此命令复用 agent-session simulator 的 OpenHands SDK 和离线 Docker 环境，需要使用该工具的 Python 环境、已配置镜像和服务商凭据。QA 运行目录须包含通过审核的题目及保存的实际生成输入。
任务 Agent 不再继承模拟对话配置中的单次输出 token 上限；保留服务商自身限制和整个任务的总 token、请求数预算。

```sh
python -m dialogue_benchmark.task_eval.run \
  --simulator-path /path/to/agent-session-simulator \
  --source-run /path/to/completed-session \
  --qa-run runs/qa-with-inputs \
  --env-file /path/to/provider.env \
  --output runs/task-pilot \
  --count 3 --task-budget 6 --revisions 5 --workers 2
```

配置的模型读取历史 QA、生成该 QA 时实际收到的材料，以及对话结束后的只读仓库，提出新需求、测试和验收标准。独立 Code Agent 实现需求，验收者检查基线、参考实现和测试质量。默认最多修正五次，保留每次失败记录。

`count` 是完成两组执行的任务目标；分批探索不同 QA 衍生的新需求，直到达标、QA 池耗尽或 `task-budget` 用完。探索上限默认是目标的两倍，失败记录全部保留。

对话结束后的代码保存为没有上游 remote 的独立 Git 基线；`--baseline` 可复用已固定的干净仓库。各 Agent 获得不含 Git 历史和其他答案的代码副本。参考实现和正式两组都保存完整代码、支持二进制的 Git 补丁及 `version.json`，记录基线提交与结果 tree。程序在临时副本中验证补丁可恢复文件新增、删除、内容和执行权限。复用时克隆基线，切到凭据中的基线提交，在该副本执行 `git apply --index /path/to/changes.patch`。

`report.html` 集中展示源 QA、实际出题输入、构造尝试、冻结测试、完整代码、补丁与两组结果。按 `input/`、`baseline/`、`qa/`、`tasks/` 组织完整运行后，执行 `python render_run.py runs/episode` 生成统一 HTML 入口和独立 QA 审阅页面。任务报告随每个任务完成更新。

`run_episode.py --source-run /path/to/completed-session --simulator-path /path/to/agent-session-simulator --env-file /path/to/provider.env --output runs/episode` 可依次执行完整流程，使用模拟器配置的模型。默认每轨目标 40 道通过题、目标 12 个完成两组评测的任务，最多探索 24 个需求；QA 并发 10，仓库任务并发 3。重叠子图先按来源合并再抽 facts；静态代码关系保留给后续 QA 组合，不重复塞入 facts 请求。

大输入的初始关系候选通过共享对象定位，并按时间线限制候选数量。只有事实进入选中的证据组，才计算其扩展选项；底层版本和代码关系继续保留。`--reuse-facts /path/to/previous-qa-run` 可复用之前的 facts，不重新调用模型抽取；要求规范化记录和分块布局完全一致，同时保留原抽取失败。复用结果的旧调用不会重复计入本次 token 消耗。

QA 完成后可用 `--resume-tasks` 复用 QA 和固定基线，从仓库任务阶段继续；启动前核对输入哈希。已有 `tasks/` 需先移到其他目录保留。参考文件通过容器用户可读的独立副本传入，原始私有文件权限保持不变。


正常完成的 Agent 先将模型原始响应和工具事件保存成经过校验的压缩轨迹，再删除其容器、快照卷和专用网络；已结束的测试也释放沙箱。镜像保留复用。中断的 Agent 和超时测试保留续跑状态，有未结束执行时暂缓整理。
完整运行入口自动整理已完成的结果；已有运行可执行 `python compact_run.py runs/episode`。保留固定基线、已接纳需求的参考实现和两组实现、补丁、冻结验收标准及测试、验收证据、统计和压缩轨迹。可解析的 QA 候选保留实际生成输入与审核理由，包括被拒绝候选。失败需求集中保留题面、验收、测试源码、测试输出和失败原因。全部执行结束且沙箱释放成功后，才删除临时代码副本、重复请求历史、批次快照和调试备份。澄清请求将共享历史合并进单个压缩审阅文件，保留每次实际问答与用量；设计探针保留可重放补丁和轨迹，不另留完整代码副本。HTML 链接压缩轨迹，不再嵌入全文副本。共享证据只存一份，子图通过引用还原。新代码副本排除 mypy/ruff 缓存，已有冻结版本的内容和哈希保持不变。
每个 Agent 另有 20 分钟执行上限，防止工具命令无限拖延；该时间仅作为资源限制，不进入评测指标。

新需求必须解决实际问题，并说明哪条历史规则会改变可观察行为。本轮主任务只采用仍适用的仓库外历史信息；仓库可轻易恢复的信息不入选，不确定的先待审核。冻结前逐条核对实际注入的 ans 是否包含有效规则、局部例外和后续纠正。缺失时修 QA 答案或缩小需求，不将私有验收说明补进 ans。

冻结验收表包含编号、要求、依据和检查方式四列。每项强制要求引用新需求或历史规则，并对应精确测试名、保存的 shell 检查或可重复的产物检查。程序保留逐测试结果，并使用 Code Agent 的项目路径配置，对参考实现和两组运行相同检查及回归命令。至少保存并重跑一个“新功能基本完成但用错历史规则”的错误补丁：基本功能检查应通过，相关历史检查应失败。

两组从同一固定代码开始，有记忆组额外收到 QA 答案。两组都可明确追问历史，响应器只根据同一冻结公开历史回答实际问题，然后继续原会话。正常完成不调用响应器；追问记录为交互次数，不扣正确性成绩。

测试项由程序直接判定，Judge 只核对事先约定的人工检查项，并引用可定位的文件和行号。历史遵守情况来自同一验收表，不再单独主观判断。明确违反要求即失败；全部强制项满足才通过；必要证据缺失则无法判断。测试已充分确定结果时不依赖 Judge 输出格式。新增反例保存后进入共同检查修订，不单独改变某一组标准。

新运行不再抽取、匹配或恢复探索 checkpoint，也不计算路线分数。报告保留所有任务的结果、逐条历史遵守、实际历史追问、开发工具调用、读取/搜索及 token。开发调用排除 think、finish，失败的实际操作仍计数。仅两组均通过时计算成本差值，`paired-differences.json` 保存“有记忆减无记忆”的差异。旧运行产物保持可读，缺失证据不补造。

## 边界

- QA 抽取只使用输入证据；仓库访问属于可选的任务构造与执行命令。
- 不提交 API key、凭据或真实运行产物。
- 模型每次只接收小证据投影，不重复发送整段对话。
- 单个块或审核失败不会删除其他独立结果。
- 自动通过只是筛选结果，不等同于人工黄金答案；本地审核文件保留证据和拒绝原因。

## 目录

`dialogue_benchmark/` 包含规范化、图构建、子图选择、模型调用、质量检查和 CLI。`tests/` 是回归测试，`examples/` 是非敏感合成输入。viewer 只是本地展示保存结果，不属于证据或记忆系统。

## 公开 episode 输入与历史约束

`run_episode.py --episode-manifest /path/to/package/manifest.json --simulator-path /path/to/agent-session-simulator --env-file /path/to/provider.env --output runs/episode` 接收 `memory-episode-v1` 包。清单指定完整公开 dialogue、截止事件、最终快照和独立运行配置，路径限定在包内，生成前核验输入与快照哈希。模型配置只记录密钥环境变量名。

`model-visible-dialogue-v1` 包含消息、工具调用和实际发送给模型的工具正文，不从私有编辑 old/new 元数据恢复完整版本。普通 QA 纳入工具证据，沿用所属讨论阶段。可重复使用 `--source-event 原始事件ID` 或 `--source-object path/to/file.py::symbol`，为两条轨道指定公开事实起点；其他事实仍可沿关系扩展。同文件或相邻时间本身不证明因果。

这类输入的需求构造会在 `history.json` 冻结有公开来源的历史事实、适用范围和替代关系，新规则只覆盖明确范围。独立验收核对来源、后续纠正、oracle 实际注入的 QA 答案和新需求可观察行为；不把构造期已完成的工作重新出题。充分历史参考证明可解性，ans 覆盖范围在冻结前独立核对。可用 `--design-probe` 单独运行无记忆设计探针，是否成功不影响入选。设计探针不产生计分路线，也不改变入选门槛。

正式两组是无记忆与直接注入 QA 答案的 oracle。两组可向同一个只读响应器询问冻结历史，在原 Code 会话中接续。响应器只回答实际问到且来源支持的信息，不看候选代码，不虚构当前环境事实。响应调用计入执行的总请求/token预算，单独保存实际回复与用量，不增加单次输出上限。当前环境查询沿用代码工具。

报告分开记录历史应用、需求正确性和成本，区分跨会话首次补问、同次获答后重复询问、对可能变化条件的合理确认。审核证据不足单列，不凭结果符合就断言记忆产生了因果收益。当前不评真实检索系统；旧 raw session 保持原输入口径可读。
