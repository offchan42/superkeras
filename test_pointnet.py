import numpy as np
import pytest

from .pointnet import build_pointnet


def test_normal_build():
    model = build_pointnet(3, 4, n_points=10)
    pred = model.predict(np.random.random((1, 10, 3)))
    assert pred.shape == (1, 10, 4)
    model = build_pointnet(2, 4, n_points=None)
    pred = model.predict(np.random.random((1, 20, 2)))
    assert pred.shape == (1, 20, 4)
    assert np.abs(1 - pred.sum(axis=-1).mean()) < 1e-5  # softmax should sum=1


def test_predict_first_last():
    with pytest.raises(ValueError):
        build_pointnet(3, 4, n_points=100, predict_first_n=10, predict_last_n=5)


def test_predict_shape():
    classes = 20
    model = build_pointnet(10, classes, n_points=32, predict_first_n=13)
    pred = model.predict(np.random.random((1, 32, 10)))
    assert pred.shape == (1, 13, classes)
    model = build_pointnet(10, classes, n_points=32, predict_last_n=13)
    pred = model.predict(np.random.random((1, 32, 10)))
    assert pred.shape == (1, 13, classes)

    # change n_points
    model = build_pointnet(10, classes, n_points=None, predict_first_n=13)
    pred = model.predict(np.random.random((1, 32, 10)))
    assert pred.shape == (1, 13, classes)
    model = build_pointnet(10, classes, n_points=None, predict_last_n=13)
    pred = model.predict(np.random.random((1, 32, 10)))
    assert pred.shape == (1, 13, classes)


def test_depth():
    model = build_pointnet(3, 4, depth=1.0)
    params1 = model.count_params()
    model = build_pointnet(3, 4, depth=0.9)
    params2 = model.count_params()
    assert params1 > params2
