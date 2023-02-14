#!/usr/bin/python3
# -*- coding: utf-8 -*-

from multiprocessing import Pool, TimeoutError
from tqdm import tqdm
from functools import partial
import json
import logging


logger = logging.getLogger(__name__)


class App:
    def __init__(self, dict_funcs=None):
        self.functions = {}
        if dict_funcs is not None:
            self.functions.update(dict_funcs)

    def add(self, key):
        def adder(func):
            self.functions[key] = func
            return func

        return adder

    def __contains__(self, item):
        return item in self.functions

    def __getitem__(self, __name: str):
        return self.functions[__name]

    def merge(self, app):
        new_app = App()
        new_app.functions = self.functions.update(app.functions)
        return new_app


def wrapper(idx_args, func):
    idx, args = idx_args
    res = func(args)
    return idx, res


def parallel_run(func, args_list, n_processes=8, initializer=None, initargs=None, **kwargs):
    idx2res = {}
    func = partial(func, **kwargs)
    n = len(args_list)
    logger.info(f"Parallel running with {n_processes} processes")
    with Pool(n_processes, initializer=initializer, initargs=initargs) as p:
        for idx, response in tqdm(p.imap_unordered(partial(wrapper, func=func),
                                                   enumerate(args_list)),
                                  total=n):
            idx2res[idx] = response

    res = [idx2res[i] for i in range(n)]
    return res


def parallel_run_timeout(func, args_list, n_processes=8, timeout=5, **kwargs):
    pool = Pool(n_processes)
    jobs = {}
    results = []
    restart = False

    for i, args in enumerate(args_list):
        jobs[i] = pool.apply_async(func, args=(args, ), kwds=kwargs)

    total_num = len(args_list)
    finished_num = 0
    fail_num = 0
    for i, r in tqdm(jobs.items()):
        try:
            finished_num += 1
            results.append(r.get(timeout=timeout))
        except TimeoutError as e:
            results.append(('exception', TimeoutError))
            logger.info("Timeout args: ")
            logger.info(args_list[i])
            fail_num += 1
            if fail_num == n_processes and total_num > finished_num:
                restart = True
                logger.info(f"All processes down, restart, remain {total_num-finished_num}/{total_num}")
                break

    pool.close()
    pool.terminate()
    pool.join()

    if restart:
        results.extend(parallel_run_timeout(func, args_list[total_num-finished_num:], n_processes, timeout, **kwargs))
    return results


def save_json(file, data_list):
    logger.info(f"Saving to {file}")
    with open(file, "w") as f:
        json.dump(data_list, f)


def load_json(file):
    logger.info(f"Loading from {file}")
    with open(file) as f:
        data = json.load(f)
    return data
