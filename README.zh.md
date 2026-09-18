# Dialogue QA Benchmark

这个仓库只做一件事：从一段对话及其对话中明确提供的代码、工具、失败和测试记录中，构建有证据依据的 QA。它独立于任何记忆产品，不连接记忆服务，也不会扫描当前仓库补充对话中没有出现的事实。

## 方法

流程把对话看成按时间排列的证据流：

1. 规范化可见消息、工具记录、代码观察、补丁、失败和测试结果。
2. 在内存中回放提供的补丁，建立轻量证据图。节点表示对话/代码对象和历史版本，边表示明确引用、修改、失败、反馈、测试和先后关系。
3. 从整张图中选择小的相关子图。相隔很远的记录只在明确指向同一个文件、函数、决定、失败或需求时组合。
4. 从小块证据中抽取带来源的事实，再只根据选中的证据生成和审核 QA。
5. 分开保存候选、审核和发布结果，方便查看被拒绝或尚未完成的题目。

这里的图只复原输入对话暴露出的信息，不等同于当前代码仓库。高价值问题应依赖过去的修改、反馈、失败、测试、决定或跨阶段对话，而不是当前仓库可以直接读出的表面事实。

支持两条题目轨道：

- 普通 QA：`single-hop`、`multi-hop`、`temporal`、`open-domain`、`adversarial`。定义参考 LoCoMo，并把不同讨论阶段作为对话中的阶段。
- 代码 QA：`fact_recall`、`history_tracking`、`behavior_inference`、`failure_diagnosis`。必须使用历史、反馈、失败/测试或跨位置关系；当前函数签名或孤立常量不能单独成题。

难度由本地静态规则计算：必要阶段/版本数量、图上的关系距离，以及是否跨越失败→修改→验证链。模型不负责决定难度。

## 安装和运行

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

每次运行写入独立目录，包含规范化证据、范围、事实、候选题、审核信息和公开 QA 视图。运行目录只供本地使用并被 Git 忽略。可以打开 `viewer/index.html`，或先运行 `viewer/build_data.py` 查看保存的运行结果。

## 边界

- 不扫描仓库，也不使用隐藏上下文回答问题。
- 不提交 API key、凭据或真实运行产物。
- 模型每次只接收小证据投影，不重复发送整段对话。
- 单个块或审核失败不会删除其他独立结果。
- 自动通过只是筛选结果，不等同于人工黄金答案；本地审核文件保留证据和拒绝原因。

## 目录

`dialogue_benchmark/` 包含规范化、图构建、子图选择、模型调用、质量检查和 CLI。`tests/` 是回归测试，`examples/` 是非敏感合成输入。viewer 只是本地展示保存结果，不属于证据或记忆系统。
