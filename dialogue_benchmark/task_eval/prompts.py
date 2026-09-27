"""Short role prompts; task content is authored by the configured model."""

from ..protocol import TASK_TYPE_GUIDANCE


def task_direction(qa_type):
    """Bind one historical purpose; the author never selects another type."""
    return "\n本题固定用途：" + TASK_TYPE_GUIDANCE[qa_type] + (
        "\nmemory-use.md 必须指出实际答案原文中的哪条知识影响新需求的哪项实现选择。"
        "只主题相关不算；用不到这条知识就写 NO_TASK.md。"
        "将对应可观察行为写入 acceptance.md，供参考验证和两组执行共同使用。\n")

SELECT_TASK = """围绕这条 QA 选择一个自然的新开发需求。
新功能要有实际用途，且答案中的某条历史信息会改变它的可观察行为。仅主题相关不够。
只使用 QA 的问题和答案确定记忆主题。history_sources 是原文索引，必要时查询；
answer_source=true 的条目是 QA 答案实际引用的来源；核对历史规则时优先读取这些条目。
原文中其他话题不能替代 QA 的主题。历史约定已确定，不等于应用它的新功能已经实现。
repository_overview 和 repository_exploration 帮你了解当前仓库，不是历史证据。
先核查相关实现、测试或文档：新功能尚未实现，至少一项必要信息无法从仓库直接恢复。
没有搜索命中不证明信息不存在；没读过的文件不能用来下结论。
不要把一次观察扩成永久政策，也不要为测试记忆凭空增加特殊条件。
每轮只返回一个决定和至多一个只读查询，不输出 JSON 或解释：
DECISION: need_evidence 或 candidate 或 stop 或 pending
REASON: 待核查事实怎样影响资格；candidate 则说明新用途、历史条件和行为影响；stop 必须有不合格证据
SOURCES: qa、已读取的 source、query 编号或 repository_exploration，逗号分隔；无引用写 none
QUERY: op|target|path|text|offset；不适用的 path 或 text 写 -；无需查询写 none
candidate 还必须给出三行：
PUBLIC_GOAL: 不含具体历史规则的一句话新功能目标
AGREEMENT_OBJECT: 历史约定涉及的对象
AGREEMENT_SCOPE: 新功能中沿用该对象的范围
END
candidate 需要实际仓库查询证据；stop 需要不合格证据；还无法判断用 pending。
查询只有 lookup/read，不执行命令。QUERY 的五段依次是操作、目标、路径、文字、偏移量。例：
QUERY: lookup|repo|.|export|0
QUERY: read|repo|src/export.py|-|0
QUERY: lookup|history|-|Maple|0
QUERY: read|history|-|source1|0
lookup 每页20项；仓库 read 的 offset 是从0开始的行号，历史 read 是字符位置。
续页使用结果的 next_offset。不得请求整份历史或重复已有完整原文。
queries 为空时不能选 candidate。需要查询只能选 need_evidence，不能同时选 candidate。
candidate/stop/pending 的 QUERY 必须为 none。
"""

HISTORY_TARGETS = """从给定的原始对话中固定与当前问题直接相关、仍可能影响后续实现的历史约定。
不要看答案要点，也不要根据当前仓库补写规则。只保留用户确认、实际观察或明确修订过的行为；
计划、猜测、一次性的操作指令不要作为约定。每条只写一个可观察的行为和适用范围。
旧规则被后续消息局部修订时，保留旧条款并在 supersedes 标明；没有可确认的规则时返回 NO_TARGETS。
输出格式：
REVIEW h1
statement: 原始对话确认的历史事实或约定
scope: 对象和适用条件
behavior: 本次未来功能中可观察的行为
sources: source1,source2
supersedes: none 或较早的 h id
END_REVIEW
只引用输入中的精确 source 别名，不输出仓库判断、答案引用或新接口设计。
"""

PUBLIC_TASK = """只根据 public_input 写一项自然的新开发需求，只输出 FILE task.md ... END_FILE。
public_input 中的历史对象和适用场景可以被提及，但其中没有历史规则；不要猜测或补写具体历史行为。
明确这次新增功能的接口、输入输出、普通兼容要求和边界；内部实现路线由开发者决定。
不要写 memory-use、验收标准、答案、历史条款、来源编号或“为了测试记忆”。
"""

PUBLIC_TASK_SIMPLE = """只根据 public_input 写一项自然的新开发需求。
沿用 public_goal 的功能目标，根据实际仓库说明接口、输入输出和兼容边界。
repository_overview、repository_exploration、repository_evidence 都是当前公开仓库资料。
选定一种适合项目的调用入口，写明模块与函数签名或完整命令格式、参数和返回形式。
出题时确定入口，不能留成“CLI 或 API 均可”，否则测试者和实现者会选择不同入口。
agreement_object 与 agreement_scope 表示要沿用的历史约定对象和范围；在需求中自然指明沿用它，
不重述或猜测该约定的具体取值和例外。仓库当前行为不能替代该外部约定。
仅约束新增能力及明确要求修改的行为；不要同时要求同一路径保持旧行为和改变旧行为。
不额外增加功能、内部实现步骤、来源编号或“为了测试记忆”。
严格只输出：
TASK
任务正文
END_TASK
不要输出 JSON、Markdown 文件头、代码围栏或其他文字。
"""

REPOSITORY_EXPLORER = """你是需求构造前的只读代码探索 Agent。当前仓库是最终对话快照。
你可以查看源码、测试、文档和配置，但不要修改候选仓库，不要生成补丁，不要运行会改变文件的命令。
围绕问题中的对象查看仓库。问题用来定位，不证明过去发生过什么；报告只写实际读到的当前仓库内容。
据此提出自然的功能扩展方向，不推测历史答案或编造历史约定。
扩展方向只写缺少的能力和可复用接口；客户的字段规则、参数取值和状态含义交由历史确认。
优先阅读入口、相关实现、相关测试和文档，确认已有能力、缺失能力、输入输出边界和可验证位置。
如果历史信息只适合做兼容约束，也要指出它会影响哪个未来行为。
完成后在 /workspace/checks/repository-exploration.md 写入以下五段，每段简短：
PROJECT_AREA:
RELEVANT_FILES:
CURRENT_CAPABILITY:
NATURAL_EXTENSION:
BOUNDARIES_AND_TESTS:
最后停止，不修改代码。
"""

PRIVATE_DRAFT = """task.md 已固定。根据固定历史目标、原始来源、实际答案和仓库查询，生成私有验收材料。
只能使用给定的历史目标，不能新增或改写历史条款；不能修改 task.md。
memory-use.md 说明每条适用目标怎样影响本任务及实际答案覆盖的内容。
history-contract.txt 必须逐条复制目标的 statement/scope/behavior/sources/supersedes，
只补 active: yes|no 和 repository: recoverable|external|uncertain。
acceptance.md 用四列表格 ID、Requirement、Basis、Check；新功能用 Basis=task，历史行为用 h id。
每项 Check 必须是具体输入、动作和预期结果，不增加任务没有要求的内部实现。
输出三个文件，文件头必须分别写成 `FILE memory-use.md`、`FILE history-contract.txt`、
`FILE acceptance.md`，文件正文后单独写 `END_FILE`；不要在 FILE 后加冒号，不要输出 task.md。
"""

PRIVATE_DRAFT_SIMPLE = """task.md 已固定。根据固定历史目标、原始来源、实际答案和仓库查询，准备私有验收。
不要修改 task.md，不要新增历史目标，不要写出内部实现路线。
先输出一段说明实际答案如何帮助本任务，格式为 USE、正文、END_USE。
然后输出验收行，格式固定为：
ACCEPT
ACCEPT a1 | task | 新功能要求 | inspect: 具体输入、动作和预期结果
ACCEPT a2 | h1 | 历史行为要求 | inspect: 具体输入、动作和预期结果
END_ACCEPT
每行只写一个可观察要求；所有行的 ID 都用 a1、a2、a3 这样的格式；新功能至少一行 task，
每个适用历史目标单独一行，并把对应的 h id 写在第二列。不要把 h1 直接写成行 ID。
不要输出 history-contract、JSON、Markdown 表格、FILE 头或其他文字。
"""

HISTORY_QUALIFY = """只核对给定的固定历史目标，不自行新增 memory_gap，也不重新写历史真值。
输入有几个目标，就必须按原顺序逐个输出几行；不能合并、遗漏或改写目标 ID。
每个目标只输出一行，格式中的占位词必须替换成下列字面值之一：
H h1 | yes | partial | sufficient | e53 | query1 | 原样引用
applicable 只能是 yes、no、uncertain（不要写 applicable）；
public 只能是 full、partial、none、uncertain（不要写 public）；
answer 只能是 sufficient、insufficient、not_applicable、uncertain。
historical_source 只能填给定目标的 source ID（例如 e53），public_source 只能填 task 或 query 编号
（例如 query1）；没有来源或引用写 none。answer_quote 必须是 historical_answer 的原文，不能改写。
最后输出一行：TASK | clean（或 leaked/uncertain）。
public 只表示题面或实际仓库是否已提供该目标；实现是否完成不影响它。
public=full 时必须引用 task 或 query；public=partial 可以引用 task/query，也可以在没有可引用查询时写 none；完全没有公开部分才写 none。
answer 只表示实际注入答案是否补足题面和仓库尚未提供的必要部分。
answer=sufficient 时，最后一列必须逐字复制 historical_answer 中的一段文字；不能从 history/source 改写或引用，不能因为来源中有同义句就填 sufficient。
如果某个目标的规则没有出现在 historical_answer 中，必须填 answer=insufficient、answer_quote=none；不能用历史来源替答案补齐。
公开需求故意不重复具体历史映射时，只要该映射会改变新增功能的可观察行为，historical_answer 对这条映射就是必要且 sufficient；不要要求题面先写出答案才承认缺口。
例如题面只说“保持字段状态一致”，答案明确说明 null 与 UNSET 的具体方向时，answer 应填 sufficient，并引用答案中的原句。
不要输出 JSON、REVIEW、END_REVIEW、解释或其他字段。
"""

DRAFT_TASK = """根据已选候选和已提供证据，一次组织草案。没有工具，不能另行调查或补造事实。
信息不足返回 REVIEW pending，reason: 缺失事实，END_REVIEW。
否则返回四个文件块，名称为 task.md、memory-use.md、history-contract.txt、acceptance.md。
格式：FILE task.md，然后换行写文件正文，最后单独一行 END_FILE。所有文件都按此格式。
正文直接换行，不用 JSON、不转义、不加引号或代码围栏。history-contract.txt 内可正常写 REVIEW。
task.md 写自然的新需求和沿用的历史对象，不重述全部历史答案、不泄露实现步骤。
memory-use.md 写实际用途、答案哪条信息改变哪个结果、仓库核查位置及剩余信息缺口。
acceptance.md 用 Markdown 四列表：ID、Requirement、Basis、Check。
ID 为 a1 等；Basis 为 task 或历史契约编号；Check 为 inspect: 具体输入、动作、预期输出。
分别列新功能和历史要求，覆盖所有有效规则。不生成测试代码。
新功能必须单列至少一行 Basis=task。历史约定在题面只引用对象，别把完整规则重新写进题面。
memory-use.md 的仓库结论必须引用实际 query 及其文件行号；未返回的内容写未知。
只在 history-contract.txt 放有历史消息来源的规则，不能把 query 编号当历史来源。
仓库文档的普通兼容要求写进 task.md，验收 Basis=task。新接口设计由新需求确定，不要求历史答案提供。
历史契约格式如下：
"""

TASK_ONLY_DRAFT = """根据已选候选、只读仓库探索和已提供仓库证据，组织一项自然的新开发需求。
本轮是 external-only QA：没有把原始对话作为任务输入，也没有可引用的历史契约。
不要臆造历史来源、历史条款或内部答案；不要把 QA 的答案内容写进公开 task.md。
输出三个文件块，名称为 task.md、memory-use.md、acceptance.md；格式为
FILE name，然后正文，最后单独一行 END_FILE。不要 JSON、代码围栏或其他文字。
task.md 只写完整的新功能、输入输出、普通兼容边界和可验证行为，不写答案、来源编号或“为了测试记忆”。
公开 task 不要写 QA 中的具体 URL、错误文本、退出码因果、具体页面名或观测结论；
这些具体外部结果只进入有记忆条件，作为诊断器的判定依据。公开 task 只保留自然的功能目标，
例如“增加文档构建日志诊断，区分外部依赖问题与源文件问题并报告页面进度”。
需求必须给出一个可执行的入口（优先使用仓库已有命令；没有时明确一个简单的脚本路径、参数和输出位置），
并写清输出中至少有哪些公开分类和成功/失败语义。不要留下“由开发者决定接口”或无法调用的抽象能力。
需求必须让 QA 答案中的至少一个具体外部结果成为可观察输出或回归条件；
不要把一次故障直接扩大成答案没有要求的修复策略、离线开关、缓存或默认行为。
memory-use.md 说明 QA 答案中的外部事实会改变哪项实现选择，但不要重述具体答案，
并说明当前仓库核查到的文件和仍需开发者确认的行为。
acceptance.md 用 Markdown 四列表：ID、Requirement、Basis、Check；所有 Basis 必须是 task，
每行一个可观察要求，至少包含一行新功能基本行为。Check 写具体输入、动作和预期输出，
Check 只用 inspect: 或精确的 test: classname::name；不要把完整 shell 命令写在 command: 后，
也不要新增 task.md 没有提出的要求。
"""

AUTHOR_TESTS = """选题资格已通过，现在完成测试构造。
先读取 /workspace/checks/task.md、memory-use.md 和 acceptance.md；这些就是本轮草案。
同目录的 task.md、memory-use.md、history-contract.txt 的内容已经固定，不能改动。
如果 /workspace/checks/history.json 不存在，本轮没有冻结历史契约，不要创建或引用 h1 等历史条款；
只为 task.md 中的公开功能写检查。
acceptance.md 只可将 Check 换成具体测试或命令，不能改 ID、Requirement、Basis 或增加要求。
如发现需求必须修改，请说明问题并结束，不自行改题。
读取 /reference/qa.json 和 /reference/qa-input.json，
后者是生成该 QA 时实际提供的材料。自主查看只读的 /workspace/candidate。
若这两个参考文件无法读取，立即报告输入错误并结束，不猜测历史，不另选主题出题。
在 /workspace/checks 实现已有验收表的检查：
| ID | Requirement | Basis | Check |
| a1 | 新功能基本行为 | task | test: test_acceptance::test_feature |
| a2 | 历史规则对应的输出 | h1 | test: test_acceptance::test_rule |
Basis 是 task 或历史契约编号，多个用逗号分隔。测试用精确 JUnit classname::name。
同一项多个测试用逗号分隔；同时引用测试和命令时用分号分隔，如 test: test_api::test_result; command: check_cli。
非 pytest 检查写 command: check_name，并保存 commands/check_name.sh，
成功返回 0、违反要求返回 1、执行异常返回 2，不依赖临时文件。
需要 Judge 的项写 inspect: 具体动作、输入和预期结果。不能只写“检查正确”。
新功能基本行为单独一行，用自动测试或固定命令验证。
task 行只检查公开功能，历史规则分别在对应 h 行检查。一个测试不要同时服务这两种行。
例如检查 JSON 可解析、记录顺序和字段顺序，不要顺便断言客户的 null 规则；后者用单独测试。
兼容性只比较约定的维度：要求键排序就检查键顺序，不能用新旧完整输出相等代替。
相关回归命令也在出题时写入 commands/*.sh，参考实现和两组一致执行。
继承执行镜像的项目路径配置，不自行假定 src 布局或修改 PYTHONPATH。
同一对象上已经要求的条件要测试组合结果，例如“持有引用 + 替换对象 + 异常退出”。
所有必须满足的行为必须在 task.md 写明或明确引用已冻结的历史契约；验收不能新增要求。
test_acceptance.py：使用 pytest，通过公开行为检查新需求，不锁定参考实现。
测试通过 candidate_root fixture 获取仓库目录；程序已提供 conftest.py，不要修改它。
新增模块或函数在测试函数内导入，不能在文件顶层导入尚不存在的接口。
基线缺少新接口应得到正常测试失败，不应阻止收集其他测试。
不要从测试文件路径推断仓库目录。需要辅助 fixture 就写在测试模块内。
若部分或全部无法稳定测试，写 TESTS_UNAVAILABLE.md 说明具体缺口，
并在 acceptance.md 写出 Judge 可执行或检查的标准，不偷偷增加需求。
先在基线上试运行测试，确认失败是因为新功能缺失；依赖和路径错误不能算。
若任务是日志解析或诊断脚本，不要运行完整 Sphinx 构建、联网构建或长时间回归；用自包含的短日志夹具覆盖公开行为，
所有探索命令应可在约 30 秒内结束，避免等待外部网络或生成整套文档。
测试构造只围绕 acceptance.md：读相关实现、写自包含检查、运行并修好测试自身错误，然后结束。
不要反复 grep 同一个词、轮询状态或为了寻找更多背景继续浏览；接口未完全确定时采用 task.md 已给出的入口，
不再回到仓库寻找第二个入口。
仓库外不要找资料。若已有验收无法实现，写 NO_TASK.md 说明具体冲突。
完成后简短总结。"""

TASK_REVIEW = """核对这个候选能否测试历史答案的帮助。先找信息缺口，再给结论。
public_task 是两组都能看到的需求；historical_answer 才是有记忆组收到的答案。
evidence.sources 是审查依据，不会额外注入给有记忆组；不能拿它替代 historical_answer。
必要历史规则若已全部写进 public_task 或仓库，选 ineligible，即使代码还未实现。
若仍有必要条件或例外未公开，写入 memory_gap，并从 historical_answer 原样摘录支持它的 answer_quote。
答案只是泛泛的“保持兼容”，却缺具体规则，选 uncertain；不能从历史原文替它补齐。
仅剩部分信息缺口时，只要求答案覆盖剩余部分；所有有效要求仍须验收。
核对历史范围、后续纠正及新需求的真实用途；扩大旧要求范围或仅主题相关选 ineligible。
仓库结论只依据 repository_queries；未读文档不能认定文档没记录，证据不足选 uncertain。
题面直接透露内部修法选 leaked；公共 API、自然行为要求和引用历史约定本身不算泄露。
全部成立才选 clean。只返回：
REVIEW
leakage: clean|leaked|ineligible|uncertain
memory_gap: 必须从历史补充的具体信息；没有写 none
answer_quote: historical_answer 中原文；没有写 none
issue: clean 时写 none，否则简短说明具体问题
END_REVIEW
"""

EXTERNAL_TASK_REVIEW = """核对这个 external-only 候选是否真的需要 QA 提供的外部观测。
public_task 是两组都能看到的自然需求；historical_answer 是只有有记忆组收到的外部观测答案。
repository_exploration 和 repository_queries 只说明当前快照，不能替代运行结果、具体错误来源、退出码因果或日志表现。
如果 public_task 或当前仓库已经完整写出答案中的具体观测，选 ineligible；不要把“仓库里没有搜到”当成外部事实。
如果需求是自然的诊断、报告或回归能力，而完成它必须依据 historical_answer 中未公开的具体观测，选 clean。
memory_gap 必须写出尚未公开、会改变实现或验收的具体观测；answer_quote 必须逐字摘自 historical_answer。
如果需求只是主题相关、只要求修复一次故障，或答案没有具体可复用观测，选 ineligible 或 uncertain。
只返回：
REVIEW
leakage: clean|ineligible|uncertain
memory_gap: 尚未公开的具体外部观测；没有写 none
answer_quote: historical_answer 中原文；没有写 none
issue: clean 时写 none，否则简短说明具体问题
END_REVIEW
"""

SOLVER = """请完成下面的开发需求。代码在 /workspace/candidate，依赖已安装，
环境无外网。自主查看仓库、修改并运行必要测试。不要修改无关行为。
若需求是日志解析或诊断脚本，不要执行 tox/uv 安装、完整 Sphinx 构建或联网等待；用短的本地日志夹具验证入口。
不要把 .tox、虚拟环境或 docs/_build 产物当作代码提交，完成后清理这类临时目录。
完成后报告实际修改和测试结果，不仅给计划。"""

HISTORY_REQUEST = """\n如果需要用户重新提供以前交代的规则，以 finish 提交且只写
HISTORY_QUESTION: 具体想确认的历史问题
系统会回答后继续同一会话。正常完成报告不要使用这个标记。两组都可追问。\n"""

VALIDATOR = """你是独立验收者。/workspace/candidate 是未修改基线；
/reference/implementation 是独立 Code Agent 实现。/reference/spec 是本轮候选需求、
验收说明和测试。不要修改需求或已有验收标准。
检查新需求确实尚未满足，以及参考实现是否真正完成。需要时在 /workspace/experiments
制作验证副本，先检查本需求的行为，再运行相关回归测试。保持仓库默认测试筛选，
不要主动启用压力、性能或长时间测试；测试不完整时亲自检查缺口并给证据。
对同一对象上可以同时发生的条件做组合验证：沿用正常行为，叠加替换、保留引用、
异常退出等需求中已有条件。测试各条件分别通过，不能代替组合检查。
将这些条件、预期结果和对应测试写到 /workspace/checks/coverage.md。
先将遗漏的组合检查写为 /workspace/checks/test_interactions.py，再实际运行。
补充文件必须自包含，不依赖临时实验文件；需要的辅助函数和数据也写在该文件内。
新增测试只能检查 task.md 已要求或明确引用的冻结历史契约，不能限定内部实现。无新增用例也写 coverage.md。
测试失败先区分测试自身错误与实现错误；确认参考实现违反需求时，保存用例、
coverage.md 和 revise 报告后结束，未完成项写 uncertain，不再继续其他检查。
无法自动测试的条件在 coverage.md 写出可重复执行的 Judge 检查步骤和期望结果；
这些文件会随需求一起冻结。只在临时实验里测过、未保存的检查不算覆盖。
检查错误实现要实际执行，不能凭代码相似判断。已发现的错误逃过检查时，
补入测试或固定检查步骤。参考实现失败则 revise。
至少保存一个“新功能基本完成，但遗漏或错用有效 external 历史规则”的错误变体。
在 /workspace/checks/m1.patch 保存相对参考实现的 git diff 补丁，
补丁路径必须是 a/项目相对路径 和 b/项目相对路径，不含 /reference 等容器前缀。
在 mutations.txt 写 REVIEW m1、acceptance: a2、END_REVIEW（各占一行）。
a2 是违反的历史验收项。补丁不可破坏新功能基本行为，程序会应用并重跑。
没有历史规则时不要求此类变体。不要改变原始基线和参考实现。
对 inspect: 项按冻结步骤检查参考实现，输出 acceptance-review.txt：
REVIEW a1
status: passed 或 failed 或 uncertain
evidence: /reference/implementation/文件:起始行-结束行，或 /workspace/checks/证据文件:起始行-结束行
END_REVIEW
每项只写一个真实证据位置，无证据写 none。
将结果写到 /workspace/checks/validation.txt，格式：
BASELINE: unmet 或 met 或 uncertain
REFERENCE: pass 或 fail 或 uncertain
TESTS: executable 或 partial 或 unavailable
MUTATIONS: caught 或 missed 或 unavailable
COVERAGE: complete 或 gaps 或 uncertain
VERDICT: accept 或 revise 或 skip
后面用简短段落给出真实执行命令、结果和需要修正的具体问题。
只有新需求未满足且参考实现已验证可行，才 accept。不能执行测试不等于不可判断；
明确依据候选标准检查代码并指出证据。所有已发现缺口均补入将冻结的检查后才写
COVERAGE: complete；仍有缺口时写 gaps 并 revise。证据不足则 uncertain/revise。
这是一次有限的验收，不要反复分段打印 checks.json 或重复 grep 同一路径。
若自动测试已给出清晰的基线/参考结果，直接保存 coverage.md、acceptance-review.txt、mutations.txt 和 validation.txt；
对 inspect 项可用一次短命令或源码行号作为证据。不要等待网络、运行完整文档构建或轮询状态。
"""

JUDGE = """你是独立验收者。只读代码在 /workspace/candidate；
冻结验收表在 /reference/spec/acceptance.md，程序执行结果在 /reference/checks.json。
只检查验收表中 inspect: 项，严格执行已冻结的动作、输入和预期结果。
不重新解释历史、不判断是否表现出记忆、不添加要求。功能没实现不能算不适用。
在 /workspace/checks/acceptance-review.txt 每项写：
REVIEW a1
status: passed 或 failed 或 uncertain
evidence: /workspace/candidate/文件:起始行-结束行，或 /workspace/checks/证据文件:起始行-结束行
END_REVIEW
evidence 只能填写一个真实文件位置；执行证据请保存为文本。无充分证据填 none。
复述历史不算行为正确；行为符合但未提历史仍通过。程序已确定的测试项不必复判。
如发现自动测试漏掉反例，保存 /workspace/checks/counterexample.md 和可重放检查，
交给共同复核，不单独修改一组标准。
"""

HISTORY_CONTRACT = """在 history-contract.txt 按事件顺序写契约，保留旧规则与后续局部修订：
REVIEW h1
statement: 一条已公开事实或约定，不写新任务的实现方案
scope: 对象和适用条件；后续修订只覆盖其明确范围
sources: 程序提供的 source1 等精确别名，或完整原始事件 id；不得自己截短
supersedes: 被此条局部替代的前面契约 id，或 none
behavior: 本任务可观察的应用行为；不能限定内部实现或固定工具路线
active: yes 或 no
repository: recoverable 或 external 或 uncertain
END_REVIEW
repository 表示当前仓库是否充分给出该信息；需要实际查看源码/测试/文档，
没搜到不等于不可恢复。memory-use.md 写核查位置和缺口。
active 表示本任务是否仍适用，scope 写清截止时仍适用的范围。
被局部替代的旧规则仅在剩余范围适用，完全失效写 no，保留来源追溯。
至少一条验收实际使用的有效规则必须 external；可以同时保留 recoverable 兼容要求。
recoverable 要给仓库约定依据，不能只说代码现在如此。uncertain 先查清，否则停止。
external 规则中未由题面和仓库提供的必要信息必须由实际 ans 覆盖；普通兼容要求仍需验收。
至少留有一项这样的信息缺口；已公开部分不要求 ans 重复。未公开设定不能写入。
acceptance.md 的行为只来自新需求和这里明确引用的历史契约。
"""

HISTORY_VALIDATOR = """\n本轮 /reference/spec/history.json 保存截止点、公开原文及历史契约。
题面明确引用的旧约定可以作为验收要求。逐条检查 statement/scope/supersedes
是否被 sources 原文支持、是否遗漏后续相关纠正；检查实际 oracle_answer 是否被原文支持。
不能把同文件或先后顺序当因果，也不能把建议当已确认约定。
核查每项规则的范围与 repository 判断，实际查看仓库，不能将隐藏的外部规则改标 recoverable。
至少一条有效 external 规则确实未被仓库和题面完整提供；否则 revise。
核对 oracle_answer 是否完整覆盖每项有效 external 规则中题面和仓库未提供的内容，包括例外和后续纠正。
在 /workspace/checks/oracle-review.txt 对每项有效 external 规则写：
REVIEW h1
coverage: complete 或 provided 或 missing 或 stale 或 uncertain
quote: complete 时为 oracle_answer 中覆盖剩余缺口的原文；provided 时为题面完整给出该规则的原文
END_REVIEW
只有题面完整给出规则才用 provided；仅给部分条件仍需 complete 并核对剩余缺口。
至少一项必须 complete 且确有缺口；全部 provided 不合格。仓库可恢复却标 external 应修正契约。
缺失就 revise 并说明需修正 QA 答案或缩小需求，不把验收说明补进 ans。
核对行为是真实新需求，不重做构造期已经完成的扩展。
严格以冻结的 task.md 和 acceptance.md 为验收范围。历史契约的行为只能检查
acceptance.md 中明确以该 h id 为 Basis 的行；不得把历史答案中的一般表述扩大到
未引用该 h id 的旧入口、其他格式或额外调用方。若 task.md 明确要求现有入口保持兼容，
历史检查不得改变这一兼容边界。
在 validation.txt 另写 HISTORY: supported 或 unsupported 或 uncertain；
不支持或遗漏有效更新就 revise。保持冻结契约不变，不临时发明历史或新增要求。
"""
CLARIFY = """只判断开发者最后的公开回复是否有尚待回答的历史/外部信息问题。
只从 supplied_history 回答实际问到的内容，遵守对象、条件和替代关系。
没有问题选 no_question；问题无法由冻结事实回答选 unavailable，不能猜当前环境，
不能主动纠错、追加任务、提供没问的事实或新修法。普通完成报告不是问题。
若要求确认可能变化的状态目前是否仍成立，只有历史记录而无当前证据时选 unavailable。
只返回：
REVIEW clarification
status: answer 或 no_question 或 unavailable
sources: 实际依据的公开事件 id，逗号分隔；没有则 none
reply: 仅 answer 时给自然回答，其余填 none
kind: historical_reask 或 same_session_repeat 或 update_confirmation 或 none
END_REVIEW
historical_reask 是跨会话重新索取已明确历史；same_session_repeat 是本次交互已答过
相同内容又问；update_confirmation 是确认可能变化的条件。类型依据 exchange，
不依据是否注入了历史答案；输出分类不给开发者。
"""
