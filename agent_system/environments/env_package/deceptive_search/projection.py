import re
from typing import List, Tuple


def _postprocess_action(action: str) -> str:
    """Trim everything after the first relevant closing tag."""
    for tag in ["</search>", "</insider>", "</answer>"]:
        if tag in action:
            return action.split(tag, 1)[0] + tag
    return action


def deceptive_search_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Project LLM actions into (results, valids) for the deceptive search env.

    Extraction priority (first match wins): <search> → <insider> → <answer>+<source>

    Validity: valids[i] = 0 when:
    - Tags of more than one type are present (mixed action).
    - Duplicate tags of the same type appear.
    - No recognisable tag found at all.
    """
    re_search = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
    re_insider = re.compile(r"<insider>(.*?)</insider>", re.IGNORECASE | re.DOTALL)
    re_answer = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
    re_source = re.compile(r"<source>(.*?)</source>", re.IGNORECASE | re.DOTALL)

    re_search_tag = re.compile(r"<search>", re.IGNORECASE)
    re_insider_tag = re.compile(r"<insider>", re.IGNORECASE)
    re_answer_tag = re.compile(r"<answer>", re.IGNORECASE)

    results: List[str] = []
    valids: List[int] = [1] * len(actions)

    for i, action in enumerate(actions):
        trimmed = _postprocess_action(action)

        # --- Extraction (priority order) ---
        m = re_search.search(trimmed)
        if m:
            results.append(f"<search>{m.group(1).strip()}</search>")
        else:
            m = re_insider.search(trimmed)
            if m:
                results.append(f"<insider>{m.group(1).strip()}</insider>")
            else:
                m = re_answer.search(trimmed)
                if m:
                    part = f"<answer>{m.group(1).strip()}</answer>"
                    # Grab <source> from the original (un-trimmed) action
                    ms = re_source.search(action)
                    if ms:
                        part += f"<source>{ms.group(1).strip()}</source>"
                    else:
                        valids[i] = 0  # <answer> without <source> is invalid
                    results.append(part)
                else:
                    results.append("")
                    valids[i] = 0

        # --- Validity checks ---
        tag_counts = [
            len(re_search_tag.findall(action)),
            len(re_insider_tag.findall(action)),
            len(re_answer_tag.findall(action)),
        ]
        # Mixed types present
        if sum(1 for c in tag_counts if c > 0) > 1:
            valids[i] = 0
        # Duplicate tags of same type
        if any(c > 1 for c in tag_counts):
            valids[i] = 0

    return results, valids
