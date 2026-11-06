import numpy as np
import pytest

from src.detector import analyse_drop, blood_group, detect
from src.generate_samples import GROUPS, clumped_drop, smooth_drop


@pytest.mark.parametrize("a,b,d,expected", [
    (True, False, True, "A+"), (False, True, False, "B-"), (True, True, True, "AB+"), (False, False, False, "O-"),
])
def test_blood_group_rules(a, b, d, expected):
    assert blood_group(a, b, d) == expected


def test_clumped_and_smooth_drops_are_told_apart():
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert analyse_drop(clumped_drop(rng)).agglutinated
        assert not analyse_drop(smooth_drop(rng)).agglutinated


@pytest.mark.parametrize("group", list(GROUPS))
def test_every_group_is_detected(group):
    rng = np.random.default_rng(list(GROUPS).index(group))
    imgs = [clumped_drop(rng) if f else smooth_drop(rng) for f in GROUPS[group]]
    found, _ = detect(*imgs)
    assert found == group
