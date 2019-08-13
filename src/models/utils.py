from functools import lru_cache


@lru_cache(maxsize=1)
def cached_model_predict(model, X):
    return model.predict(X)
