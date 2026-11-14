import math
from toolmech.config import load_config  # noqa: F401  (puts when2tool on sys.path)
from toolmech.labels import answer_key, semantic_entropy, call_position


def test_numeric_and_text_equivalence():
    assert answer_key("33") == answer_key("33.0") == answer_key(" 33 ") == answer_key("\\text{33}")
    assert answer_key("1,000") == answer_key("1000") == answer_key("1000.0")
    assert answer_key("1,000") != answer_key("[1, 0]")
    assert answer_key("\\text{Paris}") == answer_key("paris")
    assert answer_key("[1, 2, 3]") == answer_key("[1,2,3]")
    assert answer_key("[1, 2, 3]") != answer_key("[3, 2, 1]")
    assert answer_key("") == answer_key(None)
    assert answer_key("34") != answer_key("33")


def test_semantic_entropy():
    assert semantic_entropy(["33"] * 10) == (0.0, 1)
    se, n = semantic_entropy(["33"] * 5 + ["34"] * 5)
    assert n == 2 and math.isclose(se, math.log(2))
    se, n = semantic_entropy([str(i) for i in range(10)])
    assert n == 10 and math.isclose(se, math.log(10))


def test_call_position():
    assert call_position("<tool_call>\n{...}") == (True, False)
    assert call_position("  \n<tool_call>{}") == (True, False)
    assert call_position("Let me check.<tool_call>{}") == (True, True)
    assert call_position("\\boxed{33}") == (False, False)


def test_big_ints_and_int_float_merge():
    big = "1" + "0" * 400
    assert answer_key(big) == answer_key(big)
    assert answer_key(big) != answer_key("1" + "0" * 399 + "1")
    assert answer_key("33") == answer_key("33.0") == answer_key("\\text{33}") == answer_key("33.000")
    assert answer_key("2.5") == answer_key("\\text{2.5}") != answer_key("2.50001")


def test_sets_and_unhashables():
    assert answer_key("{1, 2}") == answer_key("{2, 1}")
    assert answer_key("{1, 2}") != answer_key("[1, 2]")
    assert answer_key("[{1, 2}, {3}]") == answer_key("[{2, 1}, {3}]")
    assert semantic_entropy(["{1, 2}", "{2,1}", "{3}"])[1] == 2


def test_abstain_detection_and_se_abstain():
    from toolmech.labels import is_abstain
    for a in ("unknown", "Unknown", "N/A", "cannot be determined", "not available", "?", "none"):
        assert is_abstain(a), a
    for a in ("0", "1908", "[1, 2]", "none of them were odd" * 0 + "noneworthy", ""):
        assert not is_abstain(a), a
    assert is_abstain("0", "HistoricalYearEnv") and not is_abstain("0", "CountingEnv")
    ans = ["unknown", "N/A", "0", "1441", "1441"]
    se_plain, n_plain = semantic_entropy(ans)
    se_ab, n_ab = semantic_entropy(ans, "HistoricalYearEnv", abstain_cluster=True)
    assert n_plain == 4 and n_ab == 2 and se_ab < se_plain


def test_abstain_no_false_positives_on_names():
    from toolmech.labels import is_abstain
    for a in ("banana", "Ljubljana", "Astana", "Na", "Nana", "DNA"):
        assert not is_abstain(a), a
    for a in ("N/A", "n/a", "\\text{N/A}"):
        assert is_abstain(a), a
