from tinydb import TinyDB
import yaml
from datetime import datetime


def run_benchmark(benchmark, benchmark_name, config, database, stable):
    summary = _get_summary(config, benchmark_name, stable)
    env = _get_env(config)
    params = _get_parameters(config)
    benchmark.load_parameters(**params)
    benchmark.run()
    report = benchmark.report()
    _save_run(database, summary, env, report)

def _get_summary(config_path, benchmark_name, stable):
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    now = datetime.now()
    summary = {
        "user": config["user"],
        "group": config["group"],
        "script": benchmark_name + ".py",
        "date": now.strftime("%m/%d/%y-%H:%M:%S"),
        "stable": stable
    }
    return summary

def _get_env(config_path):
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    env = {
        "system": config["system"]
    }
    return env

def _get_parameters(config_path):
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config["parameters"]

def _save_run(database, summary, env, report):
    
    db = TinyDB(database)
    
    data = {
        "summary": summary,
        "env": env,
        **report
    }
    
    db.insert(data)