"""Package the existing QA viewer and link all outputs of a local episode run."""

import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import shutil

from viewer.build_data import build


def render(root):
    root = Path(root).resolve()
    qa = root / "qa"
    viewer = root / "qa-viewer"
    viewer.mkdir(exist_ok=True)
    template = Path(__file__).parent / "viewer"
    for source in template.iterdir():
        if source.suffix in {".html", ".css", ".js"} and source.name != "data.js":
            shutil.copy2(source, viewer / source.name)
    data = build(qa)
    (viewer / "data.js").write_text("window.BENCHMARK_DATA = " +
        json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029") + ";\n")
    questions = json.loads((qa / "qa-public.json").read_text())["questions"]
    counts = Counter(q.get("qa_mode", "code") for q in questions)
    manifest = json.loads((qa / "manifest.json").read_text())
    progress = manifest.get("progress", {})
    qa_rows = []
    for mode, label in (("general", "普通题"), ("code", "代码题")):
        unique = len({q["question"].strip() for q in questions if q.get("qa_mode", "code") == mode})
        qa_rows.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            label, progress.get("targets", {}).get(mode, "—"), counts[mode], unique,
            escape(progress.get("stop_reasons", {}).get(mode, "未记录"))))
    task_manifest = root / "tasks/manifest.json"
    task_summary = "需求生成尚未完成。"
    if task_manifest.exists():
        tasks = json.loads(task_manifest.read_text())
        evaluated = [t for t in tasks.get("tasks", []) if t["status"] == "evaluated"]
        passed = sum(all(t.get("comparison", {}).get(mode, {}).get("result") == "passed"
                         for mode in ("without_memory", "with_memory")) for t in evaluated)
        task_summary = "目标 %s 个 · 已完成成对评测 %s 个 · 两组均通过 %s 个 · 状态 %s" % (
            tasks.get("target", "—"), len(evaluated), passed, tasks.get("stop_reason", "运行中"))
    baseline = json.loads((root / "baseline.json").read_text())
    receipt = json.loads((root / "input/conversion.json").read_text())
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Dialogue → QA → 新需求</title>
<style>body{font:17px/1.8 system-ui;max-width:900px;margin:60px auto;padding:0 24px;color:#19283d;background:#f5f7fb}
section{background:white;padding:24px;border-radius:14px;margin:22px 0;border:1px solid #dce3ed}a{color:#235dac}code{overflow-wrap:anywhere}
table{width:100%%;border-collapse:collapse;font-size:14px}th,td{text-align:left;border-bottom:1px solid #dce3ed;padding:8px}</style>
<h1>Dialogue → QA → 新需求</h1><p>%s</p>%s
<section><h2>QA 与证据</h2><p>普通题 %d 道 · 代码题 %d 道</p>
<table><tr><th>轨道</th><th>目标</th><th>已发布</th><th>不同题干</th><th>停止原因</th></tr>%s</table>
<p>保留原始运行结果；“不同题干”仅合并完全相同的题干，不代表语义去重或人工质量认证。</p>
<a href="qa-viewer/index.html">打开图与完整候选审阅</a> · <a href="qa/qa-public.json">最终 QA</a> ·
<a href="qa/qa-audit.json">审核记录</a> · <a href="qa/stages/">最终 QA 的实际生成输入</a></section>
<section><h2>新需求与两组代码</h2><p>模型提出新需求、生成验收标准并验证参考实现；随后分别运行无记忆和注入历史答案两组。</p>
<p>%s</p><a href="tasks/report.html">打开需求、测试、代码与轨迹对比</a> ·
<a href="tasks/report.md">Checkpoint 对照表</a> · <a href="tasks/manifest.json">任务状态</a></section>
<section><h2>固定基线</h2><p>对话结束后的代码，独立保存在本次运行中。</p><code>%s</code><p>
<a href="baseline/">基线代码</a> · <a href="baseline.json">版本凭据</a> · <a href="input/conversion.json">输入转换记录</a></p>
<p>每份结果代码都附带基于此提交的补丁和恢复校验记录。各需求互不串接。</p></section></html>''' % (
        escape("%s 条用户消息 / %s 条可见消息 / %s 条规范化记录" % (
            receipt["visible_messages"].get("user", 0),
            sum(receipt["visible_messages"].values()), receipt["records"])),
        ('<p><a href="observations.md">本轮结果与方法问题</a> · '
         '<a href="verification.json">版本和数量核验</a></p>') if (root / "observations.md").exists() else "",
        counts["general"], counts["code"], "".join(qa_rows), escape(task_summary), escape(baseline["base_commit"]))
    (root / "report.html").write_text(page, encoding="utf-8")
    print(root / "report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    render(parser.parse_args().run)
