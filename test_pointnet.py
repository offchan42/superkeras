import numpy as np
import pytest

from .pointnet import build_pointnet


def test_normal_build():
    model = build_pointnet(3, 4, n_points=10)
    pred = model.predict(np.random.random((1, 10, 3)))
    assert pred.shape == (1, 10, 4)


def test_select_first_last():
    with pytest.raises(ValueError):
        build_pointnet(3, 4, n_points=100, predict_first_n=10, predict_last_n=5)
