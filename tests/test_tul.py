import math
import torch
from toolmech.tul import windowed_tul

V, TOOL, EOS, A = 5, 3, 4, 0


def _logits(ps):
    """Per-step logits giving p(tool)=p at each step (rest mass on token A)."""
    rows = []
    for p in ps:
        l = torch.full((V,), -1e4)
        l[TOOL] = math.log(p)
        l[A] = math.log(1 - p)
        rows.append(l)
    return torch.stack(rows).unsqueeze(0)


def logit(p):
    return math.log(p) - math.log1p(-p)


def test_window_math_and_stopping():
    ps = [0.1, 0.2, 0.3, 0.4]
    gen = torch.tensor([[A, A, TOOL, A]])          # greedy path calls at step 2
    out = windowed_tul(_logits(ps), gen, [TOOL], [EOS], windows=(1, 2, 3, 4))
    assert math.isclose(out[1].item(), logit(0.1), rel_tol=1e-4)
    assert math.isclose(out[2].item(), logit(1 - 0.9 * 0.8), rel_tol=1e-4)
    p3 = 1 - 0.9 * 0.8 * 0.7
    assert math.isclose(out[3].item(), logit(p3), rel_tol=1e-4)
    assert math.isclose(out[4].item(), logit(p3), rel_tol=1e-4)   # stops at the call
    assert out["call_pos"].item() == 2 and out["path_len"].item() == 3


def test_eos_stops_window():
    gen = torch.tensor([[A, EOS, A, A]])
    out = windowed_tul(_logits([0.1, 0.1, 0.5, 0.5]), gen, [TOOL], [EOS], windows=(4,))
    assert math.isclose(out[4].item(), logit(1 - 0.9 * 0.9), rel_tol=1e-4)
    assert out["call_pos"].item() == -1


def test_saturated_probabilities_are_finite():
    gen = torch.tensor([[TOOL]])
    out = windowed_tul(_logits([1 - 1e-9]), gen, [TOOL], [EOS], windows=(1,))
    assert torch.isfinite(out[1]).all() and out[1].item() > 10
