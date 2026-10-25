import re

# [source name] or [A; B]; markdown links [text](url) are left alone
_CITATION = re.compile(r"\[([^\[\]]+)\](?!\()")
_PREFIX = re.compile(r"^\s*(source|kaynak)\s*:\s*", re.IGNORECASE)


def _key(source: str) -> str:
    return " ".join(_PREFIX.sub("", source).split()).strip(" .").casefold()


def cite(messages: list) -> tuple[str, list[dict], list[str]]:
    """Matches the [source] citations in the last answer against passages the tool actually retrieved.

    Returns: the answer with [1], [2] markers; sources in citation order; citations that matched nothing.
    """
    # Every tool result still in the thread: these are the only passages the model could have seen
    retrieved: dict[str, dict] = {}
    for m in messages:
        for hit in getattr(m, "artifact", None) or []:
            key = _key(hit["source"])
            if key not in retrieved or hit["score"] > retrieved[key]["score"]:
                retrieved[key] = hit

    sources: list[dict] = []
    numbers: dict[str, int] = {}
    unverified: list[str] = []

    def replace(match: re.Match) -> str:
        marks = []
        for part in match.group(1).split(";"):
            hit = retrieved.get(_key(part))
            if hit is None:
                unverified.append(part.strip())
                marks.append("[?]")
                continue
            if hit["doc_id"] not in numbers:
                numbers[hit["doc_id"]] = len(sources) + 1
                sources.append({**hit, "n": numbers[hit["doc_id"]]})
            marks.append(f"[{numbers[hit['doc_id']]}]")
        return "".join(marks)

    return _CITATION.sub(replace, messages[-1].content), sources, unverified


if __name__ == "__main__":
    from types import SimpleNamespace as M

    hit = {"doc_id": "d1", "source": "policies/leave-policy.md", "score": 0.8}
    messages = [
        M(type="tool", artifact=[hit, {**hit, "score": 0.5}]),
        M(
            type="ai",
            content="A [policies/leave-policy.md]. B [Source: Policies/Leave-Policy.md.]"
            " C [made-up.md; policies/leave-policy.md] [link](http://x)",
        ),
    ]
    answer, sources, unverified = cite(messages)
    assert answer == "A [1]. B [1] C [?][1] [link](http://x)", answer
    assert [(s["n"], s["score"]) for s in sources] == [(1, 0.8)]
    assert unverified == ["made-up.md"]
    print("ok")
