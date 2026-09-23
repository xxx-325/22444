"""Short role prompts; task content is authored by the configured model."""

from ..protocol import TASK_TYPE_GUIDANCE


def task_direction(qa_type):
    """Bind one historical purpose; the author never selects another type."""
    return "\n本题固定用途：" + TASK_TYPE_GUIDANCE[qa_type] + (
        "\nmemory-use.md 必须指出实际答案原文中的哪条知识影响新需求的哪项实现选择。"
        "只主题相关不算；用不到这条知识就写 NO_TASK.md。"
        "将对应可观察行为写入 acceptance.md，供参考验证和两组执行共同使用。\n")

AUTHOR = """你是出题者。读取 /reference/qa.json 和 /reference/qa-input.json，
后者是生成该 QA 时实际提供的材料。自主查看只读的 /workspace/candidate。
若这两个参考文件无法读取，立即报告输入错误并结束，不猜测历史，不另选主题出题。
先看 QA 直接关联的文件，按固定历史用途提出新开发需求；没有合适需求就写 NO_TASK.md。
从历史知识延伸一个真实、范围适中的新开发需求。基线已有的修复要保留，
新需求必须尚未完成。不要把 QA 改成重做原修复，也不要只考参数或文件名。
公开说明输入、输出、兼容要求，不泄露内部改法。历史答案只能提供旧经验。
不要把历史答案中的内部定位或修复方法复制进公开需求；必要的公共 API 名称可以写。
在 /workspace/checks 写下列文件：
task.md：给开发者的新需求，中文。
acceptance.md：逐条可判定的行为要求、对应验证方式，以及哪些必须由 Judge 判断。
同一对象上能同时发生的条件要写明组合结果，例如“持有引用 + 替换对象 + 异常退出”。
所有必须满足的行为必须在 task.md 写明或明确引用已冻结的历史契约；验收不能新增要求。
memory-use.md：一句说明历史答案可能帮助哪项开发决定，不声称已产生收益。
test_acceptance.py：使用 pytest，通过公开行为检查新需求，不锁定参考实现。
若部分或全部无法稳定测试，写 TESTS_UNAVAILABLE.md 说明具体缺口，
并在 acceptance.md 写出 Judge 可执行或检查的标准，不偷偷增加需求。
先在基线上试运行测试，确认失败是因为新功能缺失；依赖和路径错误不能算。
仓库外不要找资料。若没有合适的新需求，写 NO_TASK.md 说明原因。
完成后简短总结。"""

TASK_REVIEW = """只判断公开开发需求是否提前给出了历史答案中的解题帮助。
允许：公共 API、可观察的输入输出、触发条件和必须保留的兼容行为。
不允许：复述历史修复过程、指出负责修复的内部类/函数/分支、直接给出实现方法。
公开需求出现历史答案里的私有类/函数名，或写出其内部调用顺序，即为 leaked；
写在“背景”或“兼容要求”里也一样。兼容要求应描述对外行为，不指定内部调用顺序。
不能为了隐藏答案删掉用户必须知道的需求。仅因公共 API 或行为与答案重合，不算泄露。
题面可以明确引用已有历史约定及对象，不必复述约定内容；引用本身不是泄露。
只检查 public_task 是否泄露；historical_answer 本来就可以包含内部方法。
选 clean（没有提前给出）、leaked（提前给出）或 uncertain（不能确定）。
只返回以下文本；issue 在 clean 时为 none，否则简短指出原文位置及问题，不写新需求：
REVIEW
leakage: clean|leaked|uncertain
issue: 引用公开需求中的泄露原句并简述问题，或 none
END_REVIEW
"""

SOLVER = """请完成下面的开发需求。代码在 /workspace/candidate，依赖已安装，
环境无外网。自主查看仓库、修改并运行必要测试。不要修改无关行为。
完成后报告实际修改和测试结果，不仅给计划。"""

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
如有测试，在实验副本制作两个简单错误实现（如跳过新行为、破坏边界），验证能否识别；
不要仅凭代码相似判断。任意真实错误实现逃过检查，都必须补入上述测试或 Judge 步骤；
不能因为另外两个错误实现被抓到，就忽略这项缺口。参考实现失败则 revise。
不要改变原始基线和参考实现。
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
"""

CHECKPOINT_EXTRACT = """从一次已通过验收的无记忆参考运行中，提炼实际发生的探索动作。
只使用 task 和 steps；每步包含工具动作及真实返回，正文可能标明省略。
每项只描述一个获取信息的动作，写明具体对象和目的，例如“读取某函数的退出分支”
或“运行脚本复现某条件下的数据丢失”。不要写“理解了”“应该检查”或最优路线。
同一目的的重复查看合并，保留支持它的步骤号。cat 和 rg 只是不同工具。
不收录写代码、最终回归测试、环境准备和纯粹的失败重试；实际复现缺陷属于探索。
根据步骤先后区分：修改前复现旧问题属于探索；修改后检查改动是否生效属于验证，
即使只跑一个小脚本也不收录。修改后若出现新错误，为定位该错误而调查仍属于探索。
不可由最终代码反推探索动作，不把省略内容当作已知。不要凑数量。
只返回如下文本块；SOURCES 是输入中的步骤号，TEXT 是一个探索动作：
FACT checkpoint
SOURCES: 1,3
TEXT: 动作、对象和目的
END_FACT
没有可支持的探索动作就只返回 NO_FACTS。不要计算难度或覆盖率。
"""

CHECKPOINT_MATCH = """只核对代码 Agent 是否走过给定的探索 checkpoint。
steps 仅含该 Agent 的工具动作与返回。不要用最终代码、Judge 的动作或测试成绩推断。
对每项选择：observed（有证据完成相同信息获取动作）、alternative（有证据走了不同
探索路线替代该动作）、skipped（完整轨迹中未发生）、uncertain（证据不足）。
cat 与 rg 获取同一信息算 observed，不因工具不同判 alternative。
修改后测试不能替代修改前复现；仅运行测试不能当作阅读源码；读到代码不等于理解。
alternative 必须指出实际替代动作；仅“未做但不影响结果”应为 skipped。
返回被省略或执行结果缺失，无法确认该动作时用 uncertain。不要计算比例或解释记忆收益。
每项返回一个块，使用提供的 checkpoint 编号：
REVIEW 1
status: observed|alternative|skipped|uncertain
sources: 对应步骤号，用逗号分隔；无证据填 none
evidence: 一句实际动作和返回的说明
END_REVIEW
observed 和 alternative 必须有步骤号。不要添加清单之外的 checkpoint。
"""

JUDGE = """你是独立 Judge。只读候选代码在 /workspace/candidate，冻结的需求、
验收说明和测试在 /reference/spec。执行器测试结果在 /reference/checks.json。
先检查实现是否满足需求；测试通过只是其覆盖范围内的证据。无法自动测试的部分，
按照冻结 acceptance.md 和 coverage.md 亲自检查代码或运行验证，特别核对其中的
组合场景和无法自动测试的步骤。没有执行且不能确认的要求判 uncertain。
不要临时添加新要求。
同一确定性检查不要反复执行；不运行需求未要求的压力测试或大次数循环。
发现确定违反需求的行为后，保存失败证据和 verdict.txt 并结束。
每个判断引用实际执行结果或代码位置。你不知道是否提供了记忆，不推测组别。
在 /workspace/checks/verdict.txt 写：
RESULT: passed 或 failed 或 uncertain
后面简短列出需求完成或未完成的证据。
真实失败不能判成功；测试自身或环境故障要区分于候选实现错误。
"""

HISTORY_AUTHOR = """\n本轮使用公开历史契约。/reference/history.json 是实际公开历史，
selected_event_ids 指向 QA 相关资料；查找同对象后续纠正，不能只看旧结论。
task.md 明确新功能及其沿用的历史约定对象；不重复历史问题已经完成的修复或扩展。
具体旧规则可以通过明确引用约定来要求遵循，不必在题面重述答案。
在 history-contract.txt 按事件顺序写契约，保留旧规则与后续局部修订：
REVIEW h1
statement: 一条已公开事实或约定，不写新任务的实现方案
scope: 对象和适用条件；后续修订只覆盖其明确范围
sources: history.json 中的原始公开事件 id，逗号分隔
supersedes: 被此条局部替代的前面契约 id，或 none
behavior: 本任务可观察的应用行为；不能限定内部实现或固定工具路线
repository: recoverable 或 external 或 uncertain
END_REVIEW
repository 表示当前仓库是否充分给出该信息；需要实际查看源码/测试/文档，
没搜到不等于不可恢复。memory-use.md 写核查位置和缺口。
旧新规则不能无条件同时要求满足，scope 之外不自动覆盖。未公开设定不能写入。
acceptance.md 的行为只来自新需求和这里明确引用的历史契约。
"""

HISTORY_VALIDATOR = """\n本轮 /reference/spec/history.json 保存截止点、公开原文及历史契约。
题面明确引用的旧约定可以作为验收要求。逐条检查 statement/scope/supersedes
是否被 sources 原文支持、是否遗漏后续相关纠正；检查实际 oracle_answer 是否被原文支持。
不能把同文件或先后顺序当因果，也不能把建议当已确认约定。
充分历史参考验证可解性，不证明仅注入 oracle_answer 已足够。
核对行为是真实新需求，不重做构造期已经完成的扩展。
在 validation.txt 另写 HISTORY: supported 或 unsupported 或 uncertain；
不支持或遗漏有效更新就 revise。保持冻结契约不变，不临时发明历史或新增要求。
"""

HISTORY_JUDGE = """\n本轮另读 /reference/spec/history.json 和 /reference/trajectory.json、
/reference/clarifications.json（如有）。按截止点与对象判断历史适用范围。
在 /workspace/checks/history-review.txt 对每条契约写：
REVIEW 契约id
status: applied 或 violated 或 not_applicable 或 insufficient
evidence: 实现/工具动作/测试的具体证据；无可判定证据写 none
END_REVIEW
口头复述、检索命中或最终通过本身不能证明应用。只判断可观察的行为是否符合，
不推断内部认知或归因于记忆；失效/范围外的旧规则正确不采用是 not_applicable。
发现功能失败后也保存已查证的历史应用，其余填 insufficient，不推断全部条目。
违反仍适用的契约意味着未完成该需求，但证据不足应写 uncertain。
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
