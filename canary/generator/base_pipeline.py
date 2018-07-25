class DataTransformPipeline():
    def __init__(self, tansformers, kinds):
        self.transformers = tansformers
        self.kinds = kinds

    def transform(self, X, y=None):
        for trans in self.transformers:
            if X['kind'] in self.kinds:
                X, y = trans.transform(X, y=y)
        return X, y
