import importlib


def get_dataset_wrapper(name, **kwargs):
    return importlib.import_module('src.dataset_readers.dataset_wrappers.{}'.format(name)).DatasetWrapper(**kwargs)

