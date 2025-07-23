import importlib


def load_function(path):
    """
    Load a function from a module.
    :param path: The path to the function, e.g. "module.submodule.function".
    :return: The function object.
    """
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


class Singleton(type):

    _instance = None
    
    def __new__(cls,tasks_args):
        """Singleton pattern to ensure only one instance of PipeEngine exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance