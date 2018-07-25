from abc import ABC


class Transformer(ABC):
    def __init__(self, kinds):
        self.kinds = kinds

    def transform(self, X, y):
        if X['kind'] in self.kinds:
            raise ValueError
        raise NotImplementedError
