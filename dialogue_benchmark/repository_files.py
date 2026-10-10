"""Repository file listing and literal search without external tools.

Repository probes and task selection used to shell out to ``rg``.  A missing
binary (for example under a launcher with a reduced PATH) then failed every
probe.  These helpers keep the same visible defaults for benchmark snapshots:
hidden entries and ``.git`` are skipped, a root ``.gitignore`` applies inside
a Git repository, binary files are not searched, and results are sorted so
repeated runs see the same evidence pages.
"""

import fnmatch
import os
from pathlib import Path

_BINARY_SAMPLE = 8192
_LONG_LINE_MARKER = " [... omitted end of long line]"


def _ignore_rules(root):
    gitignore = root / ".gitignore"
    if not (root / ".git").exists() or not gitignore.is_file():
        return []
    rules = []
    for line in gitignore.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        # Negations would re-include files; listing a few extra files is the
        # conservative direction, so they are not modelled.
        if line and not line.startswith(("#", "!")):
            rules.append(line)
    return rules


def _ignored(relative, is_dir, rules):
    name = relative.rsplit("/", 1)[-1]
    for rule in rules:
        if rule.endswith("/") and not is_dir:
            continue
        pattern = rule.strip("/")
        if "/" in rule.rstrip("/"):
            if fnmatch.fnmatchcase(relative, pattern.lstrip("/")):
                return True
        elif fnmatch.fnmatchcase(name, pattern):
            return True
    return False


def list_files(root, start=None):
    """Return sorted POSIX paths, relative to ``root``, of visible files under ``start``.

    Relative results stay valid whether or not the caller resolved symlinks
    in ``root`` (for example ``/tmp`` -> ``/private/tmp`` on macOS).
    """
    root = Path(root).resolve()
    start = Path(start).resolve() if start is not None else root
    if start != root and root not in start.parents:
        raise ValueError("Listing start is outside the repository")
    if start.is_file():
        return [start.relative_to(root).as_posix()]
    rules = _ignore_rules(root)
    found = []
    for directory, names, files in os.walk(start):
        directory = Path(directory)
        relative_dir = directory.relative_to(root).as_posix()
        prefix = "" if relative_dir == "." else relative_dir + "/"
        names[:] = sorted(name for name in names if not name.startswith(".")
                          and not _ignored(prefix + name, True, rules))
        for name in files:
            if not name.startswith(".") and not _ignored(prefix + name, False, rules):
                found.append(prefix + name)
    return sorted(found)


def _is_binary(path):
    with path.open("rb") as stream:
        return b"\0" in stream.read(_BINARY_SAMPLE)


def search_text(root, start, keyword, max_columns=500):
    """Return ``path:line:text`` matches of a fixed string, paths relative to ``root``."""
    root = Path(root).resolve()
    matches = []
    for relative in list_files(root, start):
        path = root / relative
        try:
            if _is_binary(path):
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, 1):
            if keyword in line:
                text = line if len(line) <= max_columns else line[:max_columns] + _LONG_LINE_MARKER
                matches.append("%s:%d:%s" % (relative, number, text))
    return matches
