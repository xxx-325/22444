"""Strict unified diff application to in-memory content only."""

import re


def apply_diff(content, diff):
    source = content.splitlines(keepends=True)
    lines = diff.splitlines(keepends=True)
    output = []
    cursor = 0
    index = 0
    hunks = 0
    while index < len(lines):
        header = lines[index].rstrip("\r\n")
        if header.startswith(("--- ", "+++ ", "diff ", "index ")):
            index += 1
            continue
        match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", header)
        if not match:
            raise ValueError("Unsupported diff header")
        old_start, old_count, new_start, new_count = (
            int(match[1]), int(match[2] or 1), int(match[3]), int(match[4] or 1))
        start = old_start if old_count == 0 else old_start - 1
        if start < cursor or start > len(source):
            raise ValueError("Overlapping or out-of-range hunk")
        output.extend(source[cursor:start])
        expected_new = new_start if new_count == 0 else new_start - 1
        if len(output) != expected_new:
            raise ValueError("New hunk offset mismatch")
        cursor = start
        index += 1
        removed = added = 0
        while index < len(lines) and not lines[index].startswith("@@"):
            line = lines[index]
            index += 1
            if not line or line[0] not in " +-":
                raise ValueError("Unsupported diff body")
            operation, body = line[0], line[1:]
            if index < len(lines) and lines[index].startswith("\\ No newline at end of file"):
                body = body.rstrip("\r\n")
                index += 1
            if operation in " -":
                if cursor >= len(source) or source[cursor] != body:
                    raise ValueError("Patch context mismatch")
                cursor += 1
                removed += 1
            if operation in " +":
                output.append(body)
                added += 1
        if (removed, added) != (old_count, new_count):
            raise ValueError("Hunk count mismatch")
        hunks += 1
    if not hunks:
        raise ValueError("No hunks")
    output.extend(source[cursor:])
    return "".join(output)
