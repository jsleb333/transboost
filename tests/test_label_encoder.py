import pytest
import numpy as np
from transboost.label_encoder import LabelEncoder


label_encodings = {
    0:[ 1, 1],
    1:[ 1,-1],
    2:[-1, 1],
    3:[-1,-1],
}
pred_Y = np.array([
    [1,1],
    [.5,.5],
    [0,1],
    [-.1,1],
    [-2,-.01],
])
true_Y = np.array([0,0,0,2,3])

@pytest.fixture
def label_encoder():
    return LabelEncoder(label_encodings)


class TestLabelEncoder:
    def test_encode_labels(self, label_encoder):
        Y = [0,1,2,3]
        encoded_Y, W = label_encoder.encode_labels(Y)
        assert np.all(encoded_Y == np.array([[1,1],[1,-1],[-1,1],[-1,-1]]))

    def test_decode_labels(self, label_encoder):
        assert np.all(label_encoder.decode_labels(pred_Y) == true_Y)

    def test_encoding_score_quadratic(self, label_encoder):
        score = label_encoder.encoding_score(pred_Y)
        def quad_score(y, label):
            val = 0
            for f, e, w in zip(y, label_encodings[label], label_encoder.weights_matrix[label]):
                val -= w * (f - e)**2
            return val
        true_score = np.array([[quad_score(y, label) for label in [0,1,2,3]] for y in pred_Y])
        assert np.all(np.isclose(score-1, true_score))
