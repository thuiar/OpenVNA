
class BaseTrainer(object):
    def __init__(self, config) -> None:
        self.config = config

    def do_valid(self, model, dataloader):
        NotImplementedError

