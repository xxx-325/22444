"""Small test-only helpers for mandatory simple-stage responses."""


def focus_response_for(payload):
    """Return a valid focus citing a supplied local material reference."""
    references = [
        item.get("reference") for item in payload.get("material_references", [])
        if isinstance(item, dict) and isinstance(item.get("reference"), str)
    ]
    if not references:
        raise AssertionError("simple focus fixture must expose material references")
    return {"focus": {
        "text": "围绕选定事实确认一个可复用的关系",
        "sources": [references[0]],
    }}


def maybe_focus_response(prompt, payload):
    """Return the test focus response only for the dedicated focus prompt."""
    if (isinstance(prompt, str)
            and "Return exactly two lines when a grounded focus exists" in prompt):
        return focus_response_for(payload)
    return None


def relevance_response_for(payload):
    """Return a valid direct relevance response for every supplied point."""
    candidate = (payload.get("candidates") or [{}])[0]
    point_ids = [
        "A%d" % (index + 1)
        for index, point in enumerate(candidate.get("answer_points", []))
        if isinstance(point, dict)
    ] + [
        "F%d" % (index + 1)
        for index, point in enumerate(candidate.get("forbidden_points", []))
        if isinstance(point, dict)
    ]
    return {"reviews": [{
        "id": "q1",
        "review_contract": "simple_relevance_v1",
        "point_relevance": ";".join(
            point_id + "=direct" for point_id in point_ids),
    }]}


def maybe_relevance_response(prompt, payload):
    """Return the default all-direct response only for relevance prompts."""
    if (isinstance(prompt, str)
            and "simple_relevance_v1" in prompt):
        return relevance_response_for(payload)
    return None
