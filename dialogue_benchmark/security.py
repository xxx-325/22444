"""Small shared credential detector for outbound and public QA guards."""

import re


_PRIVATE_KEY = re.compile(r"-----BEGIN [^-\n]*PRIVATE KEY-----", re.I)
_BEARER = re.compile(r"\bbearer\s+[A-Za-z0-9._~-]{8,}", re.I)
_PREFIX_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?:ghp_|github_pat_|sk-|xox[baprs]-)[A-Za-z0-9_-]{16,}",
    re.I,
)
_URL_CREDENTIAL = re.compile(r"https?://[^/\s:@]+:[^/\s@]+@", re.I)
_ASSIGNMENT = re.compile(
    r"[\"']?\b(?:api[_-]?key|access[_-]?token|secret|password|passwd|token)"
    r"[\"']?\s*[:=]\s*[\"']?([^\s,\"'`]+)", re.I,
)
_NATURAL_CREDENTIAL = re.compile(
    r"\b(?:api\s*key|access\s*token|secret|password|passwd|token|凭据|密码)"
    r"\s*(?:is|为|是|叫作|叫做)\s*[\"'`]?([^\s,，。；;\"'`]+)", re.I,
)
_CREDENTIAL_LABEL = re.compile(
    r"[\"']?\b(?:api[_-]?key|access[_-]?token|secret|password|passwd|token)"
    r"[\"']?\s*[:=]", re.I,
)
_SAFE_VALUE = re.compile(
    r"^(?:none|null|nil|empty|placeholder|example|dummy|your[_-]?token|"
    r"<[^>]+>|\*{3,}|os\.environ(?:\[[^]]+\])?|\$\{?[A-Z0-9_]+\}?$)",
    re.I,
)


def credential_detected(text):
    """Return whether text contains a concrete credential-like value."""
    if not isinstance(text, str):
        return False
    if _PRIVATE_KEY.search(text) or _BEARER.search(text):
        return True
    if _PREFIX_TOKEN.search(text) or _URL_CREDENTIAL.search(text):
        return True
    for pattern in (_ASSIGNMENT, _NATURAL_CREDENTIAL):
        for match in pattern.finditer(text):
            value = match.group(1).strip().strip(".;")
            if value and not (_SAFE_VALUE.fullmatch(value)
                              or value.lower().startswith("os.environ[")):
                return True
    return False


def redact_credential_assignments(text):
    """Mark omitted assignments; the outbound guard still checks other secrets."""
    for pattern in (_ASSIGNMENT, _NATURAL_CREDENTIAL):
        text = pattern.sub("<credential assignment omitted>", text)
    # Empty prompt labels can acquire an apparent value from JSON quote escaping.
    return _CREDENTIAL_LABEL.sub("<credential label omitted>", text)
