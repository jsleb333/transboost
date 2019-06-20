from transboost.callbacks import Callback


class SacredMetricsCallback(Callback):
    """
    This class will save a complete log of the 'step' object of the CallbacksManagerIterator into a CSV. For more flexibility, define your own 'Step' class.
    """
    def __init__(self, run_info, *args, **kwargs):
        self.run_info = run_info
        super().__init__(*args, **kwargs)

    def on_step_end(self):
        step = self.manager.step.step_number
        for key, value in self.manager.step.__dict__.items():
            if key is not 'step_number':
                self.run_info.log_scalar(key, value, step)

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass
