from functools import wraps
from os import makedirs
from os.path import dirname
import pickle


def load_pickle(pickle_file):
    with open(pickle_file, "rb") as f:
        return pickle.load(f)

def pickled(pickle_file):
    def wrap(func):
        def wrapped(*args, **kwargs):
            try:
                return load_pickle(pickle_file)
            except:
                pass
            result = func(*args, **kwargs)

            makedirs(dirname(pickle_file), exist_ok=True)
            with open(pickle_file, 'wb') as pf:
                pickle.dump(result, pf, pickle.HIGHEST_PROTOCOL)
            return result
        return wrapped
    return wrap
