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

SELECT_TASK = """围绕这条 QA 选择一项尚未完成、可验收的后续业务工作。
若提供 rejected_draft，按其中具体缺口重选同一业务链内的工作；不要重复已公开全部规则的目标，也不凭空补协议。
给出 development_workflow 时，保留其业务输入、处理和交付结果，结合当前仓库细化需求。
这条链路只确定业务目标，不是历史证据；不能从中抄入具体历史取值，也不能缩减或更换交付目标。
可以复用已有接口、命令或流程完成工作；只有缺少必要能力时才新增入口。无法成立就 stop 或 pending。
答案中的某条历史信息必须改变本次处理或交付结果。仅主题相关、换函数名或包装固定配置不够。
先确定答案明确给出的规则能决定什么结果，再选择业务目标。适用对象、有效期、撤销条件等若是剩余信息缺口，业务必须实际使用它们；不要选只靠现有审批状态即可完成的工作。
作答者应依据历史应用约定，交付接口不能把待考规则作为必填输入；可将已知规则传给已有函数。开发过程中仍可向用户追问历史。
只使用 QA 的问题和答案确定记忆主题。history_sources 是原文索引，必要时查询；
answer_source=true 的条目是 QA 答案实际引用的来源；核对历史规则时优先读取这些条目。
原文中其他话题不能替代 QA 的主题。已有处理能力不等于本次业务工作已经完成。
repository_overview 和 repository_exploration 帮你了解当前仓库，不是历史证据。
先核查相关实现、测试、文档或产物：本次工作尚未完成，至少一项必要信息无法从仓库直接恢复。
没有搜索命中不证明信息不存在；没读过的文件不能用来下结论。
后续工作须保持 QA 已确认的对象、日期或周期和条件；仓库边界问题只有影响这一范围内的交付时才可作为需求，不能把特定周期约定扩成通用规则。
未确认的缺记录等边界不能当成已批准历史，也不能将内部实现方式当成历史约定。
每轮只返回一个决定和至多一个只读查询，不输出 JSON 或解释：
DECISION: need_evidence 或 candidate 或 stop 或 pending
REASON: 待核查事实怎样影响资格；candidate 则说明新用途、历史条件和行为影响；stop 必须有不合格证据
SOURCES: qa、已读取的 source、query 编号或 repository_exploration，逗号分隔；无引用写 none
QUERY: op|target|path|text|offset；不适用的 path 或 text 写 -；无需查询写 none
candidate 还必须给出三行：
PUBLIC_GOAL: 指明客户或系统、业务输入、处理和完整交付目标，只写“沿用已确认协议”及其业务范围；不得写历史答案中的字段名、取值、例外、保留/省略方向或具体兼容结果
AGREEMENT_OBJECT: 历史约定涉及的业务对象，不复述规则内容
AGREEMENT_SCOPE: 本次工作中沿用该对象的业务范围，不复述规则内容
END
candidate 需要实际仓库查询证据；stop 需要不合格证据；还无法判断用 pending。
查询只有 lookup/read，不执行命令。QUERY 的五段依次是操作、目标、路径、文字、偏移量。例：
QUERY: lookup|repo|.|export|0
QUERY: read|repo|src/export.py|-|0
QUERY: lookup|history|-|Maple|0
QUERY: read|history|-|source1|0
lookup 每页20项；仓库 read 的 offset 是从0开始的行号，历史 read 是字符位置。
续页使用结果的 next_offset。不得请求整份历史或重复已有完整原文。
已执行过的同一查询不要再次请求；零匹配只说明本次文字查找没有命中，随后应读取相关入口或文件，或者基于已有证据作出决定。
queries 为空时不能选 candidate。需要查询只能选 need_evidence，不能同时选 candidate。
candidate/stop/pending 的 QUERY 必须为 none。
"""

HISTORY_TARGETS = """从给定的原始对话中固定回答当前问题必须恢复的已确认历史约定。
question 限定对象、周期、适用范围和必要结论；agreement 只提供后续用途，不能扩大这些边界。
不要看答案要点，也不要根据当前仓库补写规则。只保留用户确认、实际观察或明确修订过的行为；
未确认的计划、猜测、缺记录等边界和普通的一次执行命令不要作为约定。
仍适用于本任务的、已确认的周期限定授权或状态必须保留，不得扩展其周期或范围。
假设例子仅解释同一规则的边界时，不单独列为目标，也不视为已发生的新批准或修正。
保留当前问题所需规则的真实适用条件、例外和已确认的局部修正。
每条只写一个可观察的行为和适用范围。
仅当后续公开消息明确修订同一对象的重叠适用范围时，保留旧条款并在 supersedes 标明。
同一消息可并列不同周期的有效规则；各周期分别适用而非替代时，supersedes 写 none，不为下一周期约定另造被替代的旧规则。没有可确认的规则时返回 NO_TARGETS。
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

PUBLIC_TASK_SIMPLE = """根据给定的业务目标和当前仓库，写一项简短、自然的后续工作需求，包含四部分：
1. 本次客户或系统的业务目标。
2. 业务输入及取得方式、需要完成的处理、交付结果的形式与位置。
3. 要求作答者沿用该客户或系统之前已确认的约定。
4. 本次涉及的既有行为应保持的兼容边界。
交付优先使用已有类型或格式；需要新结果类型时写清必要字段、结构和含义，不能只给类型名。
development_workflow 若有，保留其完整输入→处理→交付目标；public_goal 用于细化，不能把报告等交付缩减为映射包装函数。
workflow 只说明业务目标，不是历史证据；不要把其中的具体历史取值、例外或处理结论抄入公开需求。
优先复用已有能力，已有接口或命令足以完成工作时直接交付业务结果。新增可调用能力须明确一个公开入口、参数及返回值；一次性交付只需明确输入和产物。
客户等普通业务上下文可以作为参数。限定历史状态、决定或授权的日期或周期属于适用边界，公开需求必须保留。
日期可以参数化，但历史信息只在已确认周期内生效，不能随参数扩展到其他日期或周期。
historical_question 指出需从历史找回的规则；交付接口不能把这些规则作为必填输入。开发者仍可向用户追问历史，题面不禁止澄清。
作答者可以把已知规则传给已有函数，也可在实现内部保存配置；不要强制硬编码或指定规则的存储方式。
用实际项目和对象名称写正文，不把 public_goal 这样的字段名写进需求。
仓库资料用于了解项目及已有接口，不代表本次工作应采用的客户规则。
具体字段处理、参数取值和状态映射由历史约定决定，本次不要猜测或补写。
如果目标要求沿用客户以前确认的协议，只写“沿用该协议”及其业务范围；不要在公开需求中列出协议的字段、取值、例外或方向。
只说明交付行为，不指定内部实现路线；保留完成这项工作所需的衔接步骤，不添加无关目标。
严格只输出：
TASK
任务正文
END_TASK
不要输出 JSON、Markdown 文件头、代码围栏或其他文字。
"""

REPOSITORY_EXPLORER = """你是需求构造前的只读代码探索 Agent。当前仓库是最终对话快照。
你可以查看源码、测试、文档和配置，但不要修改候选仓库，不要生成补丁，不要运行会改变文件的命令。
围绕问题中的对象查看仓库。问题用来定位，不证明过去发生过什么；报告只写实际读到的当前仓库内容。
据此提出自然的后续业务方向，不推测历史答案或编造历史约定。
说明哪些工作可复用已有能力、哪些确需开发；客户的字段规则、参数取值和状态含义交由历史确认。
优先阅读入口、相关实现、相关测试、文档和产物，确认已有能力、待完成工作、输入输出边界和可验证位置。
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
历史项同时检查该规则的适用范围、例外及替换或到期条件；这些私有条件不能放进 task 行。
每项 Check 必须是具体输入、动作和预期结果，不增加任务没有要求的内部实现。
输出三个文件，文件头必须分别写成 `FILE memory-use.md`、`FILE history-contract.txt`、
`FILE acceptance.md`，文件正文后单独写 `END_FILE`；不要在 FILE 后加冒号，不要输出 task.md。
"""

PRIVATE_DRAFT_SIMPLE = """task.md 已固定。根据固定历史目标、原始来源、实际答案和仓库查询，准备私有验收。
不要修改 task.md，不要新增历史目标，不要写出内部实现路线。
先输出一段说明实际答案如何帮助本任务，格式为 USE、正文、END_USE。
然后输出验收行，格式固定为：
ACCEPT
ACCEPT a1 | task | <本次工作必须交付的一项具体结果> | inspect: <输入、动作、预期结果>
ACCEPT a2 | h1 | <该历史规则在本次工作中决定的具体行为> | inspect: <输入、动作、预期结果>
END_ACCEPT
尖括号内容全部换成实际要求。每行只写一个可观察要求，覆盖题面明确的兼容要求。
旧接口保持行为的要求只检查旧接口，不能自动变成新接口的输入不变性、异常或空输入要求。
所有行的 ID 都用 a1、a2、a3 这样的格式；本次交付至少一行 task，
每个适用历史目标单独一行，并把对应的 h id 写在第二列。不要把 h1 直接写成行 ID。
历史项同时检查该规则的适用范围、例外及替换或到期条件；这些私有条件不能放进 task 行。
历史规则应由作答者应用；验收检查交付结果，不能先把待考规则作为额外输入交给作答者。
task 行的要求和 Check 都只能使用公开信息。例如分组功能可检查分组键、记录覆盖和顺序，
不能在 task 行断言某个私有状态码属于哪组；具体取值、条件和映射只在对应 h 行验收。
不要输出 history-contract、JSON、Markdown 表格、FILE 头或其他文字。
"""

HISTORY_QUALIFY = """检查新需求是否需要给定的历史信息。输入分为三种：
public_task 和 public_repository：两组都能获得的需求与仓库信息。
private_history_targets：已经核实的历史规则，只供你核对；无记忆组看不到。
injected_answer：仅有记忆组收到的答案。不要把这两种私有材料算成公开信息。
development_workflow 若有，只说明完整业务目标，不是历史证据；需求缩减或更换其交付目标时 TASK 选 uncertain。

对每条固定规则依次判断：
applicable：完成本次工作必须用这条规则吗？yes/no/uncertain。若交付接口要求调用者把待考规则作为输入，选 no；开发者追问历史不影响适用性。
作答者依据历史选择参数并调用已有函数仍属于应用规则；客户等普通上下文参数不等于规则本身。
限定历史状态、决定或授权的日期或周期属于适用边界，公开需求必须保留。
日期可以参数化，但历史信息只在已确认周期内生效，不能随参数扩展到其他日期或周期；公开需求缺少或扩大这一边界时 TASK 选 uncertain。
若公开交付入口要求调用者传映射，正文又禁止传映射，属于题面矛盾，TASK 选 uncertain。
需求若要求新增可重复调用的能力，却未定义调用入口和参数，TASK 选 uncertain；一次性交付产物不要求新增接口。
public：仅看公开信息，具体规则已明确多少？full/partial/none/uncertain。题面从 workflow 抄入的规则也属于公开信息。
“沿用以前的约定”只指明对象，不提供具体规则；同一通用函数支持多个选项，也没有说明客户选哪个。
full 必须引用 task 或 public_repository 中的真实来源；不能引用私有规则或答案。
公开信息已给出全部必要取值与适用条件时选 full，不要求它重复历史的叙述或理由。
answer：injected_answer 是否补齐仍缺的必要信息？sufficient/insufficient/uncertain；规则不适用或已完全公开用 not_applicable。
sufficient 必须从 injected_answer 原样摘录支持文字；不能用 private_history_targets 替答案补缺项，或替公开题面与答案中未锚定的“今天／下一周期”补日期后判 sufficient。

按以下模板逐行输出，无表头。原样复制每行的 H 和规则 ID，中间保留一个空格。
只替换 | 后六列的尖括号内容，选项使用上面列出的英文值，不改写、重复或新增规则 ID。
HISTORY_QUALIFY_ROWS
引用多个答案要点时，第一点写在该行末尾，其余要点可原样用 - 开头续行。
最后一行 TASK | clean；有上述矛盾或不确定时，写 TASK | uncertain: <具体问题>。
不确定也要指出具体哪里不确定；不要输出其他解释或 JSON。
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

TASK_ONLY_DRAFT = """根据已选业务目标和仓库资料，写一项可执行的后续需求。
输入含业务目标、仓库概况和实际调查结果，不含历史答案。
保持选定客户、周期和交付范围。可以自然要求沿用已确认的约定，具体规则由后续执行者应用。
描述用户要完成的工作、普通业务输入、可调用入口、输出和必要兼容边界。
既定约定由实现者应用；入口只接收当前业务数据或观测，不能把约定值改成调用者必填的参数。
优先复用现有 API 或命令；确实缺能力时，明确新入口的路径和参数。
输入须在仓库中已有或在需求中给出，字段、类型和文件格式使用公开资料。
采用已有输出格式，或明确必要字段和结构，便于独立测试。
历史协议只控制选定业务目标涉及的结果；不额外增加含义未公开的协议字段（如另起一套前置条件或资格评分）。普通输入和输出含义由题面说明，只有具体历史规则留待恢复。
写行为与验收结果即可，不规定内部调用顺序或伪代码；引用现有接口时核对真实签名。
不要求调用者另外提交历史规则文件，不猜写未知的历史约定。
只输出 FILE task.md，换行写需求正文，最后单独一行 END_FILE。
不要 JSON、其他文件或额外解释。
"""

EXTERNAL_ACCEPTANCE = """为固定的新需求整理私有验收，不改需求。
输入是公开需求、实际注入的历史答案和仓库资料。
只输出两个文件块：FILE memory-use.md、FILE acceptance.md，每个正文后独占一行 END_FILE。
memory-use.md 说明答案中哪条有效约定影响本次什么行为；引用答案原句并保留适用范围与例外。
acceptance.md 用四列 Markdown 表：ID、Requirement、Basis、Check。ID 从 a1 开始。
公开功能行的 Basis 写 task；只有历史答案才能确定的行为行写 answer。不要混写两种依据。
若某个约定值由题面给出或由调用者作为参数传入，就不能把正确使用该值算作 answer 验收。
至少一行检查新功能基本可用，至少一行检查历史约定确实被应用；每行只有一个要求。
历史行必须来自已提供答案，不能扩大范围、添加新事实或要求模型表现出记忆。
逐项核对所有依赖历史的输出含义是否能由答案确定。答案只说明某行不得处理，不代表已定义其他行的全部资格或前置条件；缺少规则时返回 NO_TASK.md，不用参考实现补定义。
Check 用 inspect: 具体输入、动作、预期结果；测试作者随后将它替换为可执行检查。
输入格式和调用入口沿用公开需求或现有仓库，不私下新增格式。
若答案不适用于本需求，只输出 FILE NO_TASK.md、具体原因、END_FILE。
不要 JSON、代码围栏或其他文字。
"""

TEST_EXECUTION = """
执行约定：
- 测试在 /workspace/checks，项目与交付文件在 /workspace/candidate，执行工作目录是项目根。
- 程序提供 conftest.py 的 candidate_root fixture。测试函数通过参数 candidate_root 接收项目根，
  用 candidate_root / 相对路径读取业务文件，并将它传给辅助函数；不从 __file__ 或环境变量猜项目路径。
- 不修改 conftest.py；辅助 fixture 放测试模块。沿用项目路径配置，不自行设置 PYTHONPATH 或假定 src 布局。
- 新增接口在测试函数内导入，让缺功能产生用例失败，不阻断收集。使用本地输入和已安装依赖。
- task.md、memory-use.md、history-contract.txt 已冻结；acceptance.md 只改 Check 列。
- 已约定精确取值的字段或文本用精确自动断言。只要求准确描述含义的文本，将该行 Check 改为 inspect: 读取指定产物，核对具体适用对象、条件和例外；不编写匹配关键词或历史原句的测试。
- 转用 inspect 时移除该项被替代的测试断言，保留其他自动检查和冻结回归。
- 读取真实业务输入时沿用其已约定格式，保留完整字段值；不要因换解析器而丢失分隔符后的内容。
- 报告中的标签和值必须对应同一项；错误标签或互相矛盾的值不能通过。未约定固定格式时用 inspect，不以子串搜索代替语义检查。
"""

AUTHOR_TESTS = """为已确定的新需求写验收测试，然后结束。你负责读材料、写测试文件，程序随后执行。
不要运行测试、收集测试或探测环境；测试目录当前为空是正常的，先完成文件。
1. 读 /workspace/checks/task.md、acceptance.md，以及同目录已有的 history-contract.txt 或 oracle-answer.json。
   它们定义本次工作和历史规则。/workspace/candidate 是只读基线，按需读相关源码和测试。
   交付结果和所需接口由 task.md 定义，不需要在历史对话中出现；历史只确定客户的规则。
   历史资格已另行审核，本轮依据固定条款写测试，不重新调查整份对话。
   只用题面或基线已公开的接口；一次性交付直接检查产物。需要调用新能力却没有约定入口，写 NO_TASK.md 说明并结束，不搜索或猜测函数别名。
2. 在 /workspace/checks/test_acceptance.py 写 pytest 用例，检查题面行为与历史规则。
   本次交付的基本要求与客户历史规则分开测试；可检查业务产物，不限定新增接口或内部实现。
   历史规则由作答者应用，不在验收时把待考查的规则作为额外输入提供给作答者。
   兼容性只检查要求的维度；同时覆盖已要求条件的组合，不增加新要求。
   旧接口的兼容要求只约束旧接口。新接口仅要求 JSON 数据正确时，用解析后的值断言。
   未约定的编码布局、装批策略、异常类型、空输入形式和输入可变性，不添加为必过断言。
   比较输出与输入内容时，在调用前深拷贝期望值；调用后不再从输入列表或其中的对象计算期望。
   边界测试需计算输入实际位于阈值哪侧；不能仅靠注释声称跨界，或依赖未约定的编码布局。
   candidate_root 只读，只能用来读取代码、固定测试数据和已有产物；绝对不要在 candidate_root 下创建、修改或删除文件。
   测试需要临时输入、输出或 CLI 文件时，使用 pytest 提供的 tmp_path/tmpdir fixture（或 /tmp 下的临时目录），
   并把路径显式传给被测入口；不要把临时文件写到 candidate_root、/workspace/candidate 或其子目录。
3. 将 acceptance.md 的 Check 列替换成对应测试位置，ID、Requirement、Basis 保持原样。
   一项要求部分依赖人工检查时，整行 Check 用 inspect:，明确复用已通过测试并检查剩余内容。
   仅在 TESTS_UNAVAILABLE.md 写人工步骤不会触发验收；不得把未测试的部分藏在 test: 行中。
   例如 test: test_acceptance::test_feature；多个测试用逗号分隔。
   非 pytest 检查可用 command: check_name，保存 commands/check_name.sh，
   成功返回0、违反要求返回1、执行异常返回2。相关离线回归也保存为命令，供两组一致执行。
   回归用例从只读基线复制到 checks 下的子目录，命令引用这些冻结副本，
   不对参考实现或作答者可新增、修改的 tests 目录重新发现测试。
   不能自动测试的项保留 inspect: 具体动作、输入、预期结果，并写 TESTS_UNAVAILABLE.md。

测试只围绕当前验收，不寻找第二个新需求。
完成后列出写入的文件并结束；实际结果由后续执行产生。""" + TEST_EXECUTION

TEST_REPAIR = """按反馈修正已有测试。本轮只写文件，不执行命令。
requirements 是不能修改的题面和历史规则；private_memory_answer 是仅供测试作者使用的答案材料；files 是待修的测试与验收表。
repository 是实际基线源码、测试和业务数据。按实际字段与接口核对测试，不凭名称猜数据结构。
时间边界用例核对题面或历史已要求的先后关系。若“后续回复”早于其所响应的记录，先判断输入是否成立；不要把冲突场景算成实现错误，也不另加未约定的时间限制。
如果答案材料包含题面没有重述的外部规则，测试必须覆盖其可观察后果，但不要把答案写入 task.md 或公开需求。
修正反馈指出的测试缺陷。保留有依据的检查，不把测试偏好的编码、异常类或空输入形式添加为要求。
检查完整文件中的同类问题。输出的预期值在调用前独立保存；输入包含嵌套对象时深拷贝，不能共享别名。
acceptance.md 只改 Check 列内容，可用 test: 或 inspect:，其他列原样保留。现有冻结回归文件和命令不变。
输出 files 中每个文件的完整内容，格式为 FILE 文件名、换行内容、END_FILE。
不输出 JSON、代码围栏、执行结果或解释。程序会运行这些文件并再次审核。
""" + TEST_EXECUTION

TEST_FILES = """根据固定需求和完整的小型 Python 仓库写验收测试，没有工具调用。
requirements 定义本次交付及适用历史，repository 是当前代码、测试和文档。
private_memory_answer 是仅供测试作者使用的答案材料；它不能写入 task.md，也不能改变公开需求。
如果其中包含尚未写入题面的外部规则或观测，必须把对应的可观察行为加入验收测试，
使错误地忽略该规则的实现会失败；不要把答案原文或内部来源编号写入公开文件。
输出 FILE test_acceptance.py 和 FILE acceptance.md 两个完整文件，每个以 END_FILE 结束。
用顶层 def test_* 函数检查交付。
基线尚未完成交付时应是用例失败而非收集失败，不因使用已有接口而新增接口要求。
测试只检查需求约定的可观察行为，不要求某种实现路线。
旧接口的约束只测旧接口；新接口未约定的异常类型、输入不变性和编码布局不加入要求。
从调用前独立副本计算输出预期。task 行测试公开功能，历史状态码和阈值只在对应 h 或 answer 行测试。
时间用例核对题面或历史已要求的先后关系；协议开始日用例也要检查“后续回复”是否发生在已有回复之后。必要时构造满足原规则的独立业务数据，不改源快照、不另加时间规则。
顺序断言读取实际返回顺序；分类规则换一种输入排列、加入重复取值再验证，确保按位置分组不能蒙混通过。
candidate_root 是只读基线，只能读取其中的源码、文档和固定测试数据。所有临时输入、输出和 CLI 产物必须写入 pytest 的 tmp_path/tmpdir 或 /tmp，
不能写入 candidate_root、/workspace/candidate 或其子目录。
acceptance.md 只替换 Check 列，其他列原样保留。自动测试用 test: test_acceptance::函数名，含义检查用 inspect: 具体检查动作和预期含义；
原仓库 tests 已由程序冻结，旧接口回归可引用 command: existing_suite。
程序会实际运行全部用例，再检查参考实现与错用历史的变体；不要编写或声称执行结果。
一次性交付直接检查产物。需要调用新能力却没有约定入口，或固定需求有具体矛盾，仅输出 FILE NO_TASK.md、问题说明、END_FILE，不猜函数别名。
不要输出 JSON、代码围栏或其他解释。
""" + TEST_EXECUTION

TASK_REVIEW = """核对这个候选能否测试历史答案的帮助。先找信息缺口，再给结论。
给出 development_workflow 时，需求须保留完整业务交付；缩减或更换目标选 ineligible。
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
给出 development_workflow 时，需求须保留完整业务交付；缩减或更换目标选 ineligible。
public_task 是两组都能看到的自然需求；historical_answer 是有记忆组收到的历史答案。
repository_exploration 和 repository_queries 只说明当前快照，不能替代历史中明确披露的外部约定、状态或实际观测。
如果 public_task 或当前仓库已经完整写出所需历史事实，选 ineligible；不要把“仓库里没有搜到”当成外部事实。
公开需求可以指向一个此前约定的业务范围，但若直接写出答案中的具体字段取值、例外或处理方向，选 ineligible。
如果需求是尚未完成的自然业务工作，完成它必须依据 historical_answer 中题面未重述的有效规则或观测，选 clean。
memory_gap 必须写出题面与仓库缺少、会改变实现或验收的具体规则、状态或观测；answer_quote 必须逐字摘自 historical_answer。
如果需求只是主题相关、只要求复述旧结果，或答案不能改变本次处理和交付，选 ineligible 或 uncertain。
只返回：
REVIEW
leakage: clean|ineligible|uncertain
memory_gap: 尚未公开的具体外部观测；没有写 none
answer_quote: historical_answer 中原文；没有写 none
issue: clean 时写 none，否则简短说明具体问题
END_REVIEW
"""

SOLVER = """请完成下面的后续工作。代码与业务文件在 /workspace/candidate，依赖已安装，
环境无外网。自主查看仓库，复用已有能力，按需开发并交付要求的业务产物，运行必要检查。不要修改无关行为。
若需求是日志解析或诊断脚本，不要执行 tox/uv 安装、完整 Sphinx 构建或联网等待；用短的本地日志夹具验证入口。
不要把 .tox、虚拟环境或 docs/_build 产物当作代码提交，完成后清理这类临时目录。
完成后报告实际修改和测试结果，不仅给计划。"""

HISTORY_REQUEST = """\n如果需要用户重新提供以前交代的规则，以 finish 提交且只写
HISTORY_QUESTION: 具体想确认的历史问题
系统会回答后继续同一会话。正常完成报告不要使用这个标记。两组都可追问。\n"""

VALIDATOR = """审核候选需求、测试和参考实现，完成后保存审核文件并结束。
输入：/reference/spec 是固定题面、验收表和测试；如有 oracle-answer.json，它是本次唯一的私有历史答案；/reference/implementation 是参考实现；
/workspace/candidate 是基线。/reference/checks.json 已保存宿主执行的基线与参考测试结果，可直接采用。
输出都放 /workspace/checks。实验修改只放 /workspace/experiments 的副本。

按顺序完成：
1. 阅读上述输入，确认基线尚未完成本次交付，参考结果满足题面与引用的历史规则。
   测试必须依据公开功能或冻结历史，不得增加要求、限定内部实现或改变原验收。
   直接采用 checks.json 的逐测试结果；无需重新枚举测试已覆盖的输入。
2. 在 coverage.md 对应验收项说明覆盖和具体缺口。缺组合用例时写自包含的
   test_interactions.py：使用 pytest 的 test_ 函数，测试数据放在同一文件，临时副本使用 tmp_path。
   用 python -m pytest 实际运行；宿主会按相同方式执行，并核对其逐项测试结果。已有测试覆盖充分则直接说明。
   测试自身错误与实现错误分开记录。有具体错误时保存 revise 报告并结束。
3. 有 external 历史规则或 Basis=answer 的验收行时，制作一个“基本功能仍能用、但违反该历史规则”的实现副本。
   实际验证后，保存相对参考实现的 Git 补丁 m1.patch，路径为 a/项目相对路径、b/项目相对路径。
   mutations.txt 写三行：REVIEW m1、acceptance: 实际违反的验收项 ID（如 a6，不是 Markdown 行号）、END_REVIEW。
   程序还会应用补丁重跑，要求 task 行通过、指定历史行（h 编号或 answer）失败。
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

CHECKS_REVIEW = """审核每项要求的检查覆盖计划，依据已保存的需求、历史契约、测试、代码变化和自动检查执行结果。
只判断：每项要求是否有充分检查；测试是否新增了题面或有效历史中不存在的强制要求。
依据明确要求找能改变该要求结果的具体反例，不要求穷举所有非法输入。保留既有字段不等于禁止新增无害字段；每个缺口须指明违反哪条原要求，不能另加限制。
test: 与 command: 核对引用的真实执行结果及断言是否覆盖要求。
inspect: 核对固定输入、检查动作和预期结果是否具体且覆盖要求；实际检查及证据将在下一阶段完成，不能仅因没有同名测试或 pytest 结果就判 gaps。
程序只执行 acceptance.md 的 Check 列：test: 行只采用其测试结果，不会执行 TESTS_UNAVAILABLE.md 中的人工步骤。
若一行尚有必要的人工检查但 Check 仍是 test:，判 gaps，要求将整行改为 inspect: 并复用已有测试结果。
仅写“由 Judge 判断”等泛泛安排，缺少具体检查动作或预期含义，仍判 gaps。
测试通过不证明测试合理。检查边界、作用范围及公开功能与历史条件的区分。
核对输入解析是否保留真实记录的完整字段；丢弃分隔符后的内容属于 gaps。报告检查若错误标签含正确标签的子串就能通过，或矛盾值仍能通过，也判 gaps。
既有行为只在题面要求兼容的范围内约束新功能；额外限定内部实现或未要求的输出布局属于 unsupported。
旧接口兼容不自动约束新接口的编码、装批、异常类型、空输入形式或输入可变性。
核对每个断言在新接口要求中的依据；这些行为未约定时，不因参考实现恰好满足就通过。
已约定精确取值的字符串或机器字段可作等值断言；仅要求准确描述含义时，不得要求与历史原句逐字相等，应以 inspect 核对固定产物与预期含义。
顺序断言须检查实际输出；若只按输入位置返回固定结果也能通过分类测试，或断言恒真，判 gaps。
边界用例应实际跨过所称阈值；仅在未约定的编码布局下才恰好位于边界的用例属于 unsupported。
回归命令应执行冻结的基线测试副本，不能发现参考实现或作答者自己新增的测试。
已有执行结果可以直接采用，不提出重复探测。输入缺少判断所需代码则 uncertain。
每项验收行返回一块，最后用 tests 检查额外测试是否同样有依据：
REVIEW a1
coverage: complete 或 gaps 或 unsupported 或 uncertain
evidence: 引用具体测试、命令或 inspect 步骤及其要求依据，或说明一项具体缺口
END_REVIEW
"""

HISTORY_MUTATION = """制作一个历史规则错误补丁并保存，然后结束。
/reference/spec 是已审核的需求、历史契约和测试，/reference/implementation 是已通过的参考交付（代码或业务产物）。
从 memory-use.md 的实际信息缺口选择错误点：违反仓库与题面未提供的必要条件，而非仅修改仓库已明确的普通取值。
把参考实现复制到 /workspace/experiments/mutant，只改这个副本，使一条 external 规则被误用，
公开交付要求仍满足，不改 QA、历史来源、私有验收材料或测试。保存相对参考实现的 /workspace/checks/m1.patch，使用 a/相对路径、b/相对路径。
再保存 /workspace/checks/mutations.txt：
REVIEW m1
acceptance: 该补丁违反的历史验收项 ID（如 a6，不是 Markdown 行号）
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

MUTATION_FILES = """故意把参考交付中的一条 external 历史规则应用错误，制作测试用的错误变体。
从 memory-use.md 的实际信息缺口选择错误点：违反仓库与题面未提供的必要条件，而非仅修改仓库已明确的普通取值。
选择 reference_sources 中本次改动的代码或业务产物（如报告），使一项历史条件对应的行为或结果出错。
公开交付要求仍满足，不改 QA、历史来源、私有验收材料或测试。
before.txt 原样摘取只出现一次的文本；after.txt 写不同的错误文本。
程序会精确替换、生成补丁，并验证公开功能行通过而指定历史行失败。只输出这三个文件：
FILE mutations.txt
REVIEW m1
acceptance: 被违反的历史验收项 ID（如 a6，不是 Markdown 行号）
file: reference_sources 中的实际路径
END_REVIEW
END_FILE
FILE before.txt
要替换的原文本
END_FILE
FILE after.txt
替换后的错误文本
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

HISTORY_SOURCE_REVIEW = """只核对固定历史规则的来源、范围及截止时的有效性。全部需核对的原文已在输入中。
对每条 active 规则，判断公开原文是否支持其 statement、scope 和 supersedes，包括截止前的纠正。
建议不等于确认事实；局部替代不扩大到其他对象或周期。不要反转新旧关系或沿用已失效的范围。
原文明确支持选 supported，明确冲突选 unsupported，范围或有效性无法确认选 uncertain。
不要求历史曾实现未来接口，也不审查答案覆盖、代码实现或验收方式。
只返回每条 active 规则的一个文本块：
REVIEW h1
support: supported 或 unsupported 或 uncertain
issue: none 或一个具体问题
END_REVIEW
unsupported 或 uncertain 必须写明原文支持上的具体问题。不要输出 coverage 或 quote。
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
