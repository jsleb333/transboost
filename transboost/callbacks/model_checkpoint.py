import os
import pickle as pkl
from copy import copy

from transboost.callbacks import PeriodicSaveCallback, PickleSaveMixin, Callback


class ModelCheckpoint(Callback):
    """
    This class makes a checkpoint of the QuadBoost model in a two separate pickle files, which can be loaded later.
    """
    def __init__(self, filename, dirname='.', resume_fit=False, *args, **kwargs):
        """
        Args:
            filename (str): The name of the file to be saved. This string can be formated using the 'format_filename' method if it is overloaded in inherited classes.
            dirname (str, optional): The name of the directory. By default, it saves in the save directory as the script.
            resume_fit (bool): Whether or not the checkpoint should resume the checkpoint of a model or start from fresh.

        By default, all files will be overwritten at each save. However, one can insert a '{round}' substring in the specified 'filename' that will be formatted with the round number before being saved to differenciate the files.
        """
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.dirname = dirname
        os.makedirs(dirname, exist_ok=True)
        self.resume_fit = resume_fit

    @property
    def filedir(self):
        return os.path.join(self.dirname, self.format_filename(self.filename))

    def format_filename(self, filename):
        return filename.format(round=self.manager.step.step_number+1)

    def dump_model(self):
        with open(self.filedir+'.model.ckpt', mode='wb') as model_file:
            shallow_copy_of_model = copy(self.manager.caller)
            del shallow_copy_of_model.weak_predictors
            del shallow_copy_of_model.weak_predictors_weights
            pkl.dump(shallow_copy_of_model, model_file)

    def dump_update(self):
        with open(self.filedir+'.update.ckpt', mode='ab') as update_file:
            update = (self.manager.caller.weak_predictors[-1],
                    self.manager.caller.weak_predictors_weights[-1])
            pkl.dump(update, update_file)

    def on_iteration_begin(self):
        self.old_filedir = self.filedir
        if not self.resume_fit:
            if os.path.exists(self.filedir):
                os.remove(self.filedir)

    def on_step_end(self):
        self.rename_old_update_save()
        self.dump_update()

        self.dump_model()
        self.erase_old_model_save()

    def rename_old_update_save(self):
        if self.filedir != self.old_filedir:
            try:
                os.rename(self.old_filedir+'.update.ckpt', self.filedir+'.update.ckpt')
            except FileNotFoundError: pass

    def erase_old_model_save(self):
        if self.filedir != self.old_filedir:
            try:
                os.remove(self.old_filedir + '.model.ckpt')
            except FileNotFoundError: pass

        self.old_filedir = self.filedir

    @staticmethod
    def load_model(filename, dirname='.'):
        with open(os.path.join(dirname, filename + '.model.ckpt'), 'rb') as model_file:
            model = pkl.load(model_file)
            model.weak_predictors = []
            model.weak_predictors_weights = []

        with open(os.path.join(dirname, filename + '.update.ckpt'), 'rb') as update_file:
            update_file.seek(-1,2)     # go to the file end.
            end_of_file = update_file.tell()   # get the end of file location
            update_file.seek(0,0)
            while update_file.tell() < end_of_file:
                (wp, wpw) = pkl.load(update_file)
                model.weak_predictors.append(wp)
                model.weak_predictors_weights.append(wpw)

        return model
