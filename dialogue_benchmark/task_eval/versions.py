"""Pin dialogue-end code and export replayable changes without sharing agent history."""

from pathlib import Path
import shutil
import subprocess
import tempfile

from .artifacts import copy_tree, fingerprint, save


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), "-c", "core.hooksPath=/dev/null",
         "-c", "core.autocrlf=false", "-c", "core.filemode=true", *args],
        check=True, capture_output=True,
    ).stdout


def pin_baseline(root):
    """Create an independent local repository from an already copied snapshot."""
    root = Path(root)
    if (root / ".git").exists():
        raise ValueError("Baseline already has Git metadata")
    git(root, "init", "--template=", "--initial-branch=baseline")
    git(root, "add", "--force", "--all")
    git(root, "-c", "user.name=Dialogue Benchmark", "-c",
        "user.email=benchmark@localhost", "commit", "--no-gpg-sign", "--allow-empty",
        "-m", "Pin dialogue-end code snapshot")
    return baseline_version(root)


def baseline_version(root):
    root = Path(root)
    if not (root / ".git").is_dir() or git(root, "status", "--porcelain"):
        raise ValueError("Expected an independent, clean baseline repository")
    return {"base_commit": git(root, "rev-parse", "HEAD").decode().strip(),
            "base_tree": git(root, "rev-parse", "HEAD^{tree}").decode().strip(),
            "content_sha256": fingerprint(root)}


def export_change(base, candidate, output):
    """Export binary/mode-aware Git patch and verify replay in a disposable clone."""
    base, candidate, output = Path(base).resolve(), Path(candidate).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    base_commit = git(base, "rev-parse", "HEAD").decode().strip()
    with tempfile.TemporaryDirectory(prefix="dialogue-patch-") as directory:
        repo = Path(directory) / "result"
        git(base, "clone", "--quiet", "--no-local", str(base), str(repo))
        # Only replace files inside this function's disposable clone.
        for path in repo.iterdir():
            if path.name == ".git":
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        clean = Path(directory) / "candidate"
        copy_tree(candidate, clean, include_caches=True)
        shutil.copytree(clean, repo, dirs_exist_ok=True)
        git(repo, "add", "--force", "--all")
        patch = git(repo, "diff", "--cached", "--binary", "--full-index",
                    "--no-ext-diff", "--no-textconv", base_commit, "--")
        changed = git(repo, "diff", "--cached", "--name-only", "-z", base_commit).decode().split("\0")
        result_tree = git(repo, "write-tree").decode().strip()
        patch_path = output / "changes.patch"
        patch_path.write_bytes(patch)
        replay = Path(directory) / "replay"
        git(base, "clone", "--quiet", "--no-local", str(base), str(replay))
        if patch:
            git(replay, "apply", "--index", "--binary", str(patch_path.resolve()))
        replay_tree = git(replay, "write-tree").decode().strip()
        if replay_tree != result_tree:
            raise ValueError("Exported patch did not reproduce the result tree")
    receipt = {"base_commit": base_commit, "result_tree": result_tree,
               "patch": "changes.patch", "replay_verified": True,
               "candidate_sha256": fingerprint(candidate),
               "changed_files": [name for name in changed if name]}
    save(output / "version.json", receipt)
    return receipt
