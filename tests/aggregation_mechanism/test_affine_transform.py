import pytest
import numpy as np

from transboost.aggregation_mechanism import AffineTransform, RandomAffine, random_affine


rotation, scale, shear, translation, center = .25, (.2, .15), (.1,.05), (4,4), (6,6)
aff_mat = np.array([
    [0.18787454, -0.04432803,  9.13872093],
    [0.06857956,  0.14330047,  8.72871979],
    [0,0,1]
])

class TestAffineTransform:
    def test_affine_matrix(self):
        aff = AffineTransform(rotation, scale, shear, translation, center).affine_matrix
        assert np.isclose(aff, aff_mat).all()

    def test_determinant(self):
        det = AffineTransform(rotation, scale, shear, translation, center).determinant
        assert np.isclose(det, aff_mat[0,0]*aff_mat[1,1] - aff_mat[1,0]*aff_mat[0,1])

    def test_call(self):
        aff = AffineTransform(rotation=np.pi/2, center=(1,1))
        X = np.arange(9).reshape(3,3) + 1
        transformed_X = np.array([
            [7,4,1],
            [8,5,2],
            [9,6,3]
        ])
        assert np.isclose(aff(X), transformed_X).all()
