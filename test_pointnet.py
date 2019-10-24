import numpy as np
import pytest

from .pointnet import build_pointnet, copy_pointnet


def test_normal_build():
    model = build_pointnet(3, 4, n_points=10, depth=0.2)
    pred = model.predict(np.random.random((1, 10, 3)))
    assert pred.shape == (1, 10, 4)
    model = build_pointnet(2, 4, n_points=None, depth=0.2)
    pred = model.predict(np.random.random((1, 20, 2)))
    assert pred.shape == (1, 20, 4)
    assert np.abs(1 - pred.sum(axis=-1).mean()) < 1e-5  # softmax should sum=1


def test_predict_first_last():
    with pytest.raises(ValueError):
        build_pointnet(
            3, 4, n_points=100, predict_first_n=10, predict_last_n=5, depth=0.1
        )


def test_predict_shape():
    classes = 20
    model = build_pointnet(10, classes, n_points=32, predict_first_n=13, depth=0.1)
    pred = model.predict(np.random.random((1, 32, 10)))
    assert pred.shape == (1, 13, classes)
    model = build_pointnet(10, classes, n_points=32, predict_last_n=13, depth=0.1)
    pred = model.predict(np.random.random((1, 32, 10)))
    assert pred.shape == (1, 13, classes)

    # change n_points
    model = build_pointnet(10, classes, n_points=None, predict_first_n=13, depth=0.1)
    pred = model.predict(np.random.random((1, 32, 10)))
    assert pred.shape == (1, 13, classes)
    model = build_pointnet(10, classes, n_points=None, predict_last_n=13, depth=0.1)
    pred = model.predict(np.random.random((1, 32, 10)))
    assert pred.shape == (1, 13, classes)


def test_depth():
    model = build_pointnet(3, 4, depth=0.2)
    params1 = model.count_params()
    model = build_pointnet(3, 4, depth=0.1)
    params2 = model.count_params()
    assert params1 > params2


def test_copy():
    depth = 0.2
    model = build_pointnet(3, 4, n_points=20, depth=depth)
    model2 = copy_pointnet(model, depth, n_points=None)
    x = np.random.random((1, 20, 3))
    y1 = model.predict(x)
    y2 = model2.predict(x)
    diff = y1 - y2
    # check that their predictions are the same
    assert np.abs(diff).max() == 0.0

    # check that first model cannot predict variable length, but 2nd model can
    with pytest.raises(ValueError):
        model.predict(np.random.random((1, 5, 3)))
    model2.predict(np.random.random((1, 5, 3)))
