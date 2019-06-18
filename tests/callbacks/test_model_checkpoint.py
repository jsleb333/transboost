import pytest
import pickle as pkl
from unittest.mock import MagicMock
import os

from transboost.callbacks import ModelCheckpoint, CallbacksManagerIterator


class DummyModel:
    def __init__(self, model_param=1):
        self.model_param = model_param
        self.weak_predictors = []

    def update_model(self, value):
        self.weak_predictors.append(value)

class TestModelCheckpoint:
    def setup_method(self, mtd):
        self.filename = 'test{round}'
        self.model_path = './test0.model.ckpt'
        self.update_path = './test0.update.ckpt'
        self.model = DummyModel()
        self.manager = CallbacksManagerIterator(caller=self.model)
        self.ckpt = ModelCheckpoint(self.filename, manager=self.manager)

    def teardown_method(self, mtd):
        del self.model
        del self.manager
        del self.ckpt
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.update_path):
            os.remove(self.update_path)

    def test_dump_model(self):
        self.model.model_param = 3
        self.ckpt.dump_model()

        assert os.path.exists(self.model_path)
        with open(self.model_path, 'rb') as model_file:
            loaded_model = pkl.load(model_file)

        assert not hasattr(loaded_model, 'weak_predictors')
        assert loaded_model.model_param == 3

    def test_dump_update(self):
        self.model.model_param = 3
        self.model.update_model(4)
        self.ckpt.dump_update()

        assert os.path.exists(self.update_path)
        with open(self.update_path, 'rb') as update_file:
            loaded_update = pkl.load(update_file)

        assert loaded_update == 4

    def test_erase_old_model_save(self):
        self.ckpt.on_iteration_begin()
        self.ckpt.dump_model()
        next(self.ckpt.manager.step)

        self.model.model_param = 3
        self.ckpt.dump_model()
        self.ckpt.erase_old_model_save()

        assert not os.path.exists('./test0.model.ckpt')
        assert os.path.exists('./test1.model.ckpt')

        os.remove('./test1.model.ckpt')

    def test_rename_old_update_save(self):
        self.ckpt.on_iteration_begin()
        self.model.update_model(4)
        self.ckpt.dump_update()

        assert os.path.exists('./test0.update.ckpt')

        next(self.ckpt.manager.step)
        self.model.update_model(4)
        self.ckpt.rename_old_update_save()
        self.ckpt.dump_update()

        assert not os.path.exists('./test0.update.ckpt')
        assert os.path.exists('./test1.update.ckpt')

        os.remove('./test1.update.ckpt')

    def test_load_model(self):
        self.ckpt.on_iteration_begin()
        self.model.update_model(4)
        self.ckpt.on_step_end()

        next(self.ckpt.manager.step)
        self.model.update_model(4)
        self.ckpt.on_step_end()

        loaded_model = ModelCheckpoint.load_model('test1')
        assert loaded_model.weak_predictors == [4,4]

        os.remove('./test1.model.ckpt')
        os.remove('./test1.update.ckpt')
