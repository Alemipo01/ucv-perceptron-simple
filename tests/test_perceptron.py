from src.dataset import training_data
from src.perceptron import Perceptron


def build_model():
    model = Perceptron()
    model.train(training_data)
    return model


def test_prediction():
    assert build_model().predict([8, 1]) == 1


def test_rejection():
    assert build_model().predict([1, 0]) == 0

