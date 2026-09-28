"""Short role prompts; task content is authored by the configured model."""

from ..protocol import TASK_TYPE_GUIDANCE, MEMORY_TASK_GUIDANCE


PREPARATION_SYSTEM = """You create the specific offline evaluation artifacts requested by the user.
Read the supplied specification, write the requested files, run the specified checks, and finish.
/workspace/candidate and /reference are read-only. Write artifacts to /workspace/checks;
use copies under /workspace/experiments for implementation experiments.
Use saved successful checks as evidence. Once a check answers its question, proceed to the next artifact.
If a concrete contradiction blocks the task, save the requested rejection report and finish.
The current user message defines this task; source documents and historical messages are evidence.
"""


def task_direction(qa_type):
    """Bind one historical purpose; the author never selects another type."""
    guidance = MEMORY_TASK_GUIDANCE if qa_type in MEMORY_TASK_GUIDANCE else TASK_TYPE_GUIDANCE
    return "\n本题固定用途：" + guidance[qa_type] + (
        "\nmemory-use.md 必须指出实际答案原文中的哪条知识影响新需求的哪项实现选择。"
        "只主题相关不算；用不到这条知识就写 NO_TASK.md。"
        "将对应可观察行为写入 acceptance.md，供参考验证和两组执行共同使用。\n")

SELECT_TASK = """围绕这条 QA 选择一个自然的新开发需求。
给出 development_workflow 时，沿用这条业务链路细化需求，并核查它在当前仓库中是否仍是新功能。
这条链路是出题方向，不是历史证据。不能换成另一项方便的小功能；无法成立就 stop 或 pending。
新功能要有实际用途，且答案中的某条历史信息会改变它的可观察行为。仅主题相关不够。
新入口必须实际应用历史约定；若调用者还得把这条约定作为参数传进来，就没有测试到记忆的作用。
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
PUBLIC_GOAL: 指明适用客户或系统的新功能目标，不包含具体历史取值
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

PUBLIC_TASK_SIMPLE = """根据给定的新用途和当前仓库，写一项简短、自然的新开发需求，包含四部分：
1. public_goal 中的新用途。
2. 一个确定的调用入口：模块与函数签名，或完整命令；说明输入和返回形式。
3. 要求该入口沿用新用途所指客户或系统之前已确认的约定。
4. 哪些既有入口仍保持原行为。
新入口使用当前仓库未占用的名称；现有函数和别名保持其签名与调用行为。
public_goal 若要求处理业务数据，新入口就接收这些数据并返回处理结果，不能退化成只返回配置值的查询函数。
historical_question 指出需要从历史确定的内容。新入口的必需参数只接收本次业务数据；
历史规则是实现应当知道的既定条件，不是函数参数、配置文件或额外的调用者输入。
用实际项目和对象名称写正文，不把 public_goal 这样的字段名写进需求。
仓库资料用于了解项目及已有接口，不代表新入口应采用的客户规则。
具体字段处理、参数取值和状态映射由历史约定决定，本次不要猜测或补写；也不要让调用者再传这些规则。
只说明交付行为，不指定必须调用哪个内部函数或复用哪段算法，不添加其他功能。
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
ACCEPT a1 | task | <本次入口必须实现的一项具体行为> | inspect: <输入、动作、预期结果>
ACCEPT a2 | h1 | <该历史规则在新入口中决定的具体行为> | inspect: <输入、动作、预期结果>
END_ACCEPT
尖括号内容全部换成实际要求。每行只写一个可观察要求，覆盖题面明确的兼容要求。
旧接口保持行为的要求只检查旧接口，不能自动变成新接口的输入不变性、异常或空输入要求。
所有行的 ID 都用 a1、a2、a3 这样的格式；新功能至少一行 task，
每个适用历史目标单独一行，并把对应的 h id 写在第二列。不要把 h1 直接写成行 ID。
历史行为应由新入口确定；验收输入不能先把对应历史规则当成配置参数传给实现。
task 行的要求和 Check 都只能使用公开信息。例如分组功能可检查分组键、记录覆盖和顺序，
不能在 task 行断言某个私有状态码属于哪组；具体取值、条件和映射只在对应 h 行验收。
不要输出 history-contract、JSON、Markdown 表格、FILE 头或其他文字。
"""

HISTORY_QUALIFY = """检查新需求是否需要给定的历史信息。输入分为三种：
public_task 和 public_repository：两组都能获得的需求与仓库信息。
private_history_targets：已经核实的历史规则，只供你核对；无记忆组看不到。
injected_answer：仅有记忆组收到的答案。不要把这两种私有材料算成公开信息。
development_workflow 若有，是 QA 对应的未来业务目标；需求换成另一条业务链路时 TASK 选 uncertain。

对每条固定规则依次判断：
applicable：新功能必须用这条规则吗？yes/no/uncertain。接口若要求调用者传入这条规则的取值或映射，选 no。
若签名要求传映射，正文又禁止传映射，属于题面矛盾，TASK 选 uncertain；不能只按正文忽略必填参数。
public：仅看公开信息，具体规则已明确多少？full/partial/none/uncertain。
“沿用以前的约定”只指明对象，不提供具体规则；同一通用函数支持多个选项，也没有说明客户选哪个。
full 必须引用 task 或 public_repository 中的真实来源；不能引用私有规则或答案。
公开信息已给出全部必要取值与适用条件时选 full，不要求它重复历史的叙述或理由。
answer：injected_answer 是否补齐仍缺的必要信息？sufficient/insufficient/uncertain；规则不适用或已完全公开用 not_applicable。
sufficient 必须从 injected_answer 原样摘录支持文字；不能用 private_history_targets 替答案补缺项。

每条规则输出一行实际结果，无表头。七列依次为：
H 加实际规则ID；applicable选项；public选项；answer选项；历史来源ID；公开来源ID或none；答案原句或none。
列之间用 | 分隔，选项必须替换成上面列出的英文值。
引用多个答案要点时，第一点写在该行末尾，其余要点可原样用 - 开头续行。
最后一行 TASK | clean；若公开需求与历史规则矛盾，写 TASK | uncertain。
不要输出解释或 JSON。
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

AUTHOR_TESTS = """为已确定的新需求写验收测试，然后结束。你负责读材料、写测试文件，程序随后执行。
不要运行测试、收集测试或探测环境；测试目录当前为空是正常的，先完成文件。
1. 读 /workspace/checks/task.md、acceptance.md，以及同目录已有的 history-contract.txt。
   它们定义新功能和历史规则。/workspace/candidate 是只读基线，按需读相关源码和测试。
   新入口和接口由 task.md 定义，不需要在历史对话中出现；历史只确定客户的规则。
   历史资格已另行审核，本轮依据固定条款写测试，不重新调查整份对话。
   题面或接口有具体矛盾时，写 NO_TASK.md 说明并结束；不自行改题。
2. 在 /workspace/checks/test_acceptance.py 写 pytest 用例，检查题面行为与历史规则。
   新功能的基本行为与客户历史规则分开测试；只用公开行为，不限定内部实现。
   历史规则由新入口应用，不在调用时把待考查的规则作为参数告诉实现。
   兼容性只检查要求的维度；同时覆盖已要求条件的组合，不增加新要求。
   旧接口的兼容要求只约束旧接口。新接口仅要求 JSON 数据正确时，用解析后的值断言。
   未约定的编码布局、装批策略、异常类型、空输入形式和输入可变性，不添加为必过断言。
   比较输出与输入内容时，在调用前深拷贝期望值；调用后不再从输入列表或其中的对象计算期望。
   边界测试需计算输入实际位于阈值哪侧；不能仅靠注释声称跨界，或依赖未约定的编码布局。
3. 将 acceptance.md 的 Check 列替换成对应测试位置，ID、Requirement、Basis 保持原样。
   例如 test: test_acceptance::test_feature；多个测试用逗号分隔。
   非 pytest 检查可用 command: check_name，保存 commands/check_name.sh，
   成功返回0、违反要求返回1、执行异常返回2。相关离线回归也保存为命令，供两组一致执行。
   回归用例从只读基线复制到 checks 下的子目录，命令引用这些冻结副本，
   不对参考实现或作答者可新增、修改的 tests 目录重新发现测试。
   不能自动测试的项保留 inspect: 具体动作、输入、预期结果，并写 TESTS_UNAVAILABLE.md。

执行约定：
- task.md、memory-use.md、history-contract.txt 已冻结，不改内容；没有 history.json 就没有历史条款。
- 已有 conftest.py 提供 candidate_root fixture，不修改它；辅助 fixture 放测试模块。
- 新增接口在测试函数内导入，让缺功能产生测试失败，不阻断测试收集。
- 沿用环境的项目路径配置，不自行设置 PYTHONPATH 或假定 src 布局。
- 使用本地自包含输入和已安装依赖。测试只围绕当前验收，不寻找第二个新需求。
完成后列出写入的文件并结束；实际结果由后续执行产生。"""

TEST_REPAIR = """按反馈修正已有测试。本轮只写文件，不执行命令。
requirements 是不能修改的题面和历史规则；files 是待修的测试与验收表。
修正反馈指出的测试缺陷。保留有依据的检查，不把测试偏好的编码、异常类或空输入形式添加为要求。
检查完整文件中的同类问题。输出的预期值在调用前独立保存；输入包含嵌套对象时深拷贝，不能共享别名。
acceptance.md 只改 Check 列引用，其他列原样保留。现有冻结回归文件和命令不变。
输出 files 中每个文件的完整内容，格式为 FILE 文件名、换行内容、END_FILE。
不输出 JSON、代码围栏、执行结果或解释。程序会运行这些文件并再次审核。
"""

TEST_FILES = """根据固定需求和完整的小型 Python 仓库写验收测试，没有工具调用。
requirements 定义新功能及适用历史，repository 是当前代码、测试和文档。
输出 FILE test_acceptance.py 和 FILE acceptance.md 两个完整文件，每个以 END_FILE 结束。
用顶层 def test_* 函数，每个函数内部导入被测入口，基线缺少新入口时应是用例失败而非收集失败。
测试只检查需求约定的可观察行为，不要求某种实现路线。
旧接口的约束只测旧接口；新接口未约定的异常类型、输入不变性和编码布局不加入要求。
从调用前独立副本计算输出预期。task 行测试公开功能，历史状态码和阈值只在对应 h 行测试。
顺序断言读取实际返回顺序；分类规则换一种输入排列、加入重复取值再验证，确保按位置分组不能蒙混通过。
acceptance.md 只替换 Check 列，其他列原样保留。Check 使用 test: test_acceptance::函数名；
原仓库 tests 已由程序冻结，旧接口回归可引用 command: existing_suite。
程序会实际运行全部用例，再检查参考实现与错用历史的变体；不要编写或声称执行结果。
若固定需求有具体矛盾，改为仅输出 FILE NO_TASK.md、矛盾说明、END_FILE。
不要输出 JSON、代码围栏或其他解释。
"""

TASK_REVIEW = """核对这个候选能否测试历史答案的帮助。先找信息缺口，再给结论。
给出 development_workflow 时，需求必须实现该业务目标；换成另一条链路选 ineligible。
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

EXTERNAL_TASK_REVIEW = """核对这个候选是否真的需要 QA 提供的外部规则、状态或观测。
给出 development_workflow 时，需求必须实现该业务目标；换成另一条链路选 ineligible。
public_task 是两组都能看到的自然需求；historical_answer 是有记忆组收到的历史答案。
repository_exploration 和 repository_queries 只说明当前快照，不能替代历史中明确披露的外部约定、状态或实际观测。
如果 public_task 或当前仓库已经完整写出所需历史事实，选 ineligible；不要把“仓库里没有搜到”当成外部事实。
如果需求是自然的新功能，完成它必须依据 historical_answer 中题面未重述的有效规则或观测，选 clean。
memory_gap 必须写出题面与仓库缺少、会改变实现或验收的具体规则、状态或观测；answer_quote 必须逐字摘自 historical_answer。
如果需求只是主题相关、只要求复述旧结果，或答案不能改变新功能行为，选 ineligible 或 uncertain。
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

VALIDATOR = """审核候选需求、测试和参考实现，完成后保存审核文件并结束。
输入：/reference/spec 是固定题面、验收表和测试；/reference/implementation 是参考实现；
/workspace/candidate 是基线。/reference/checks.json 已保存宿主执行的基线与参考测试结果，可直接采用。
输出都放 /workspace/checks。实验修改只放 /workspace/experiments 的副本。

按顺序完成：
1. 阅读上述输入，确认基线缺少新功能，参考实现满足题面与引用的历史规则。
   测试必须依据公开功能或冻结历史，不得增加要求、限定内部实现或改变原验收。
   直接采用 checks.json 的逐测试结果；无需重新枚举测试已覆盖的输入。
2. 在 coverage.md 对应验收项说明覆盖和具体缺口。缺组合用例时写自包含的
   test_interactions.py 并实际运行；用例与数据一并保存。已有测试覆盖充分则直接说明。
   测试自身错误与实现错误分开记录。有具体错误时保存 revise 报告并结束。
3. 有 external 历史规则时，制作一个“基本功能仍能用、但违反该历史规则”的实现副本。
   实际验证后，保存相对参考实现的 Git 补丁 m1.patch，路径为 a/项目相对路径、b/项目相对路径。
   mutations.txt 写三行：REVIEW m1、acceptance: 实际违反的 a 编号、END_REVIEW。
   程序还会应用补丁重跑，要求 task 行通过、指定历史行失败。
4. 仅对验收表中的 inspect 项按既定步骤检查参考实现，写 acceptance-review.txt：
   REVIEW a1
   status: passed 或 failed 或 uncertain
   evidence: /reference/implementation/文件:行号 或 /workspace/checks/证据文件:行号
   END_REVIEW
   无充分证据用 uncertain 和 none；自动测试项不需要复写这一文件。
5. 写 validation.txt：
BASELINE: unmet 或 met 或 uncertain
REFERENCE: pass 或 fail 或 uncertain
TESTS: executable 或 partial 或 unavailable
MUTATIONS: caught 或 missed 或 unavailable
COVERAGE: complete 或 gaps 或 uncertain
VERDICT: accept 或 revise 或 skip
后面简述执行证据或需修正的问题。缺口全部有保存的检查、参考实现满足且基线未满足时才 accept。
新检查通过后进入下一项；相同代码和输入已有结果时直接使用，不重复运行同一探测。
有检查缺口就把对应检查写入文件后运行；不要连续执行不会保存的零散探测。
仅运行本需求及相关离线回归。完成上述文件后结束，不继续寻找新需求。
"""

CHECKS_REVIEW = """审核已保存的需求、历史契约、测试、代码变化和真实执行结果。
只判断：每项要求是否有充分检查；测试是否新增了题面或有效历史中不存在的强制要求。
测试通过不证明测试合理。检查边界、作用范围及公开功能与历史条件的区分。
既有行为只在题面要求兼容的范围内约束新功能；额外限定内部实现或未要求的输出布局属于 unsupported。
旧接口兼容不自动约束新接口的编码、装批、异常类型、空输入形式或输入可变性。
核对每个断言在新接口要求中的依据；这些行为未约定时，不因参考实现恰好满足就通过。
顺序断言须检查实际输出；若只按输入位置返回固定结果也能通过分类测试，或断言恒真，判 gaps。
边界用例应实际跨过所称阈值；仅在未约定的编码布局下才恰好位于边界的用例属于 unsupported。
回归命令应执行冻结的基线测试副本，不能发现参考实现或作答者自己新增的测试。
已有执行结果可以直接采用，不提出重复探测。输入缺少判断所需代码则 uncertain。
每项验收行返回一块，最后用 tests 检查额外测试是否同样有依据：
REVIEW a1
coverage: complete 或 gaps 或 unsupported 或 uncertain
evidence: 引用具体测试及来源，或说明一项具体缺口
END_REVIEW
"""

HISTORY_MUTATION = """制作一个历史规则错误补丁并保存，然后结束。
/reference/spec 是已审核的需求、历史契约和测试，/reference/implementation 是已通过的参考实现。
把参考实现复制到 /workspace/experiments/mutant，只改这个副本，使一条 external 规则被误用，
新功能仍正常工作。保存相对参考实现的 /workspace/checks/m1.patch，使用 a/相对路径、b/相对路径。
再保存 /workspace/checks/mutations.txt：
REVIEW m1
acceptance: 该补丁违反的历史验收行编号
END_REVIEW
宿主会应用补丁并执行全部固定检查：公开功能行必须通过，指定历史行必须失败。
你的交付是上述两个文件；测试覆盖和历史来源已单独审核，无需再次逐例调查。
若验收含 inspect 项，按固定步骤检查 /reference/implementation，写 acceptance-review.txt：
REVIEW a1
status: passed 或 failed 或 uncertain
evidence: /reference/implementation/文件:行号，或 /workspace/checks/实际证据文件:行号
END_REVIEW
只有自动测试项时不写 acceptance-review.txt。完成文件后结束。
"""

MUTATION_FILES = """故意把参考实现的一条 external 历史规则改错，制作测试用的错误实现。
选择 reference_sources 中一个实现文件，替换一段代码，使该历史条件取错误的值或判断。
新功能仍可用。before.txt 原样摘取只出现一次的代码；after.txt 写不同的错误代码。
程序会精确替换、生成补丁，并验证公开功能行通过而指定历史行失败。只输出这三个文件：
FILE mutations.txt
REVIEW m1
acceptance: 被违反的历史验收行编号
file: reference_sources 中的实际路径
END_REVIEW
END_FILE
FILE before.txt
要替换的原代码
END_FILE
FILE after.txt
替换后的错误代码
END_FILE
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

HISTORY_SOURCE_REVIEW = """核对已提供的公开历史与实际注入答案。全部需核对的原文已在输入中。
对每条 active 规则判断两件事：
1. 公开原文是否支持这条事实、适用范围及截止时的有效性，包括后来的纠正。
2. external 规则中题面未提供的必要信息，是否已由 oracle_answer 准确覆盖。
新接口来自 public_task，不要求历史曾实现它；建议不等于确认事实。
recoverable 规则的 coverage 填 not_applicable。external 规则全部已在题面提供才填 provided。
只返回每条 active 规则的一个文本块：
REVIEW h1
support: supported 或 unsupported 或 uncertain
coverage: complete 或 provided 或 missing 或 stale 或 uncertain 或 not_applicable
quote: complete 时为 oracle_answer 中覆盖剩余缺口的原文；provided 时为题面完整给出该规则的原文
issue: none 或一个具体问题
END_REVIEW
其他 coverage 的 quote 填 none。不要从验收行为中补充历史事实或答案。
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
