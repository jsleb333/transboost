import pytest
import os, sys
sys.path.append(os.getcwd())

from transboost.callbacks import ModelPickler, ModelCheckpoint


class DummyModel:
    def __init__(self, model_param=1):
        self.model_param = model_param
        self.update_model = []

    def update_model(self, value):
        self.model_updates.append(value)

class TestModelPickler:
    @classmethod
    def setup_method(cls, mtd):
        mp = ModelPickler('test.pkl', '.')
    @classmethod
    def teardown_method(cls, mtd):
        if os.path.exists('./test.pkl'):
            os.remove('test.pkl')

    def test_save_model(self):
        mp
