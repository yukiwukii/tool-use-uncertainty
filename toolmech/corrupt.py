"""E5: corrupted tool outputs of the same type and similar length (PLAN.md E5 step 2).

`corrupt(res, key, rng, pool)` returns (new_res, info). It changes only the payload
of a successful tool result (the field the answer is read from) and keeps every other
field and the JSON shape:

  number / numeric string -> one digit changed (not the leading digit, so the
                             magnitude and length stay the same)
  bool                    -> negated
  list / dict             -> one leaf inside it corrupted
  text without digits     -> swapped for the same field from a different call of the
                             same tool (e.g. the passage for a different entity from the
                             retriever or the year lookup); a weekday -> another weekday
Mirrored fields (CodeExecutor stdout == result) are kept consistent. Failed results
are returned unchanged with info["corrupted"] = False.
"""
import copy
import json
import re

PAYLOAD_KEYS = ("result", "decoded", "encoded", "hash", "passage", "current_list", "content",
                "last_result", "hits", "stdout", "data")
MIRRORS = (("result", "stdout"),)
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_DIG = re.compile(r"\d")


def _find_payload(res):
    """(container, key) of the payload field, breadth-first through dicts."""
    q = [res]
    while q:
        d = q.pop(0)
        if not isinstance(d, dict):
            continue
        for k in PAYLOAD_KEYS:
            if k in d and d[k] not in (None, "", [], {}):
                if k == "data" and isinstance(d[k], dict):
                    q.append(d[k])
                    break
                return d, k
        q += [v for v in d.values() if isinstance(v, dict)]
    return None, None


def _change_digit(s, rng):
    pos = [m.start() for m in _DIG.finditer(s)]
    if not pos:
        return None
    lead = pos[0]
    cand = [p for p in pos if p != lead] or pos
    p = cand[int(rng.integers(len(cand)))]
    old = s[p]
    choices = [c for c in "0123456789" if c != old and not (p == lead and c == "0" and len(pos) > 1)]
    return s[:p] + choices[int(rng.integers(len(choices)))] + s[p + 1:]


def _corrupt_value(v, rng, pool_vals):
    if isinstance(v, bool):
        return not v, "negate"
    if isinstance(v, (int, float)):
        s = _change_digit(repr(v), rng)
        return (type(v)(float(s)) if isinstance(v, float) else int(s)), "digit"
    if isinstance(v, str):
        if v in WEEKDAYS:
            return WEEKDAYS[(WEEKDAYS.index(v) + 1 + int(rng.integers(6))) % 7], "weekday"
        others = [o for o in pool_vals if isinstance(o, str) and o != v]
        if not _DIG.search(v) or (len(v) > 60 and others):
            if others:   # same field from another call; prefer similar length
                others.sort(key=lambda o: abs(len(o) - len(v)))
                return others[int(rng.integers(min(3, len(others))))], "swap"
            if _DIG.search(v) is None:
                w = v.split()
                if len(w) > 1:
                    i, j = rng.choice(len(w), 2, replace=False)
                    w[i], w[j] = w[j], w[i]
                    return " ".join(w), "word_swap"
                return v[::-1], "reverse"
        return _change_digit(v, rng), "digit"
    if isinstance(v, list):
        others = [o for o in pool_vals if isinstance(o, list) and o != v]
        if v and all(isinstance(x, dict) for x in v):      # retriever hits: another query's hits
            if others:
                return others[int(rng.integers(len(others)))], "swap"
        leaves = [i for i, x in enumerate(v) if isinstance(x, (int, float, str, list, dict)) and x != []]
        if not leaves:
            return (others[0], "swap") if others else (v, "none")
        i = leaves[int(rng.integers(len(leaves)))]
        nv = list(copy.deepcopy(v))
        nv[i], how = _corrupt_value(nv[i], rng, [])
        return nv, f"list[{how}]"
    if isinstance(v, dict):
        # every numeric leaf (except counts), so whichever statistic the task asks for changes
        keys = [k for k, x in v.items() if isinstance(x, (int, float)) and not isinstance(x, bool)
                and k not in ("count", "n")]
        if not keys:
            keys = [k for k, x in v.items() if isinstance(x, (str, list, bool))][:1] or list(v)[:1]
        nv = copy.deepcopy(v)
        for k in keys:
            nv[k], _ = _corrupt_value(nv[k], rng, [])
        return nv, f"dict[{len(keys)} leaves]"
    return v, "none"


def corrupt(res, pool_key, rng, pool):
    """pool: {pool_key: [payload values seen in real runs]} (pool_key = env/tool/field)."""
    info = {"corrupted": False, "how": None, "field": None}
    if not isinstance(res, dict) or not res.get("success"):
        return res, info
    new = copy.deepcopy(res)
    d, k = _find_payload(new)
    if d is None:
        return res, info
    old = d[k]
    d[k], how = _corrupt_value(old, rng, pool.get(f"{pool_key}/{k}", []))
    for a, b in MIRRORS:
        if k == a and b in d and d[b] == (old if isinstance(old, str) else str(old)):
            d[b] = d[k] if isinstance(d[k], str) else str(d[k])
    info.update(corrupted=d[k] != old, how=how, field=k,
                len_ratio=len(json.dumps(new, ensure_ascii=False)) / max(1, len(json.dumps(res, ensure_ascii=False))))
    return new, info


def payload(res):
    d, k = _find_payload(res) if isinstance(res, dict) else (None, None)
    return (k, d[k]) if d is not None else (None, None)
