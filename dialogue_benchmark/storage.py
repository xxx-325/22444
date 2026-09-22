"""Lossless shared evidence records and compressed local traces."""

import gzip
import hashlib
import json
from pathlib import Path


def load(path):
    path = Path(path)
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        path = path.with_suffix(path.suffix + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    if isinstance(value, dict) and value.get("format") == "shared-evidence-v1":
        pool = load(path.parent / "evidence-records.json")

        def expand(item):
            if isinstance(item, dict):
                if set(item) == {"$evidence"}:
                    return pool[item["$evidence"]]
                return {key: expand(child) for key, child in item.items()}
            if isinstance(item, list):
                return [expand(child) for child in item]
            return item

        return expand(value["items"])
    return value


def save_projection(path, value):
    """Store repeated source/fact rows once, including distinct fragment variants."""
    path = Path(path)
    pool_path = path.parent / "evidence-records.json"
    pool = load(pool_path) if pool_path.exists() else {}

    def pack(item, field=None):
        if isinstance(item, list):
            if field in {"dialogue", "events", "versions", "facts"}:
                refs = []
                for row in item:
                    body = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    key = hashlib.sha256(body.encode()).hexdigest()
                    pool.setdefault(key, row)
                    refs.append({"$evidence": key})
                return refs
            return [pack(child) for child in item]
        if isinstance(item, dict):
            return {key: pack(child, key) for key, child in item.items()}
        return item

    result = {"format": "shared-evidence-v1", "items": pack(value)}
    for target, data in ((pool_path, pool), (path, result)):
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        temp.chmod(0o600)
        temp.replace(target)


def compress_file(path):
    """Verify compressed bytes before removing an uncompressed artifact."""
    path = Path(path)
    target = path.with_suffix(path.suffix + ".gz")
    temp = target.with_suffix(target.suffix + ".tmp")
    before = hashlib.sha256()
    with path.open("rb") as source, gzip.open(temp, "wb") as output:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            before.update(block)
            output.write(block)
    after = hashlib.sha256()
    with gzip.open(temp, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            after.update(block)
    if before.digest() != after.digest():
        raise ValueError("Compressed artifact verification failed: " + str(path))
    temp.chmod(0o600)
    temp.replace(target)
    path.unlink()
    return target
