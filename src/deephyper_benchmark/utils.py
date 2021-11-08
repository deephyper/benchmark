import importlib
import logging
import os
import sys
from datetime import datetime

import yaml
from tinydb import TinyDB

from deephyper_benchmark.benchmark import Benchmark

logger = logging.getLogger(__name__)


def run_config(config, verbose):
    benchmark, benchmark_name = _get_benchmark(config, verbose)
    summary = _get_summary(config, benchmark_name)
    env = _get_env(config)
    params = _get_parameters(config)
    parameters = benchmark.load_parameters(params)
    results = benchmark.run()
    _save_run(config, summary, env, parameters, results)


def _get_benchmark(config_path, verbose):
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    script = config["script"].split('/')
    src = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    script = os.path.join(src, "scripts", *script)

    logger.info(f"Loading benchmark from: {script}")

    sys.path.insert(0, os.path.dirname(script))
    module_name = os.path.basename(script)[:-3]
    mscript = importlib.import_module(module_name)

    def is_valid_benchmark(cls):
        try:
            b = cls is not Benchmark and issubclass(cls, Benchmark)
        except:
            b = False
        return b

    benchmark_classes = [
        getattr(mscript, attr)
        for attr in dir(mscript)
        if is_valid_benchmark(getattr(mscript, attr))
    ]

    assert len(
        benchmark_classes) == 1, f"Only 1 benchmark per script! Found {len(benchmark_classes)}..."

    benchmark_class = benchmark_classes[0]
    benchmark = benchmark_class(verbose=verbose)

    l_script = script.split("/")
    i_src = l_script.index("src")
    folder = os.path.join(*l_script[i_src+2:-1])
    benchmark_name = os.path.join(folder, mscript.__name__)

    return benchmark, benchmark_name


def _get_summary(config_path, benchmark_name):
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    now = datetime.now()
    summary = {
        **config["summary"],
        "script": benchmark_name + ".py",
        "date": now.strftime("%m/%d/%y-%H:%M:%S")
    }
    return summary


def _get_env(config_path):
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config["env"]


def _get_parameters(config_path):
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config["parameters"]


def _save_run(config_path, summary, env, parameters, results):
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    db = TinyDB(config["database"])

    data = {
        "summary": summary,
        "env": env,
        "parameters": parameters,
        "results": results
    }

    db.insert(data)
