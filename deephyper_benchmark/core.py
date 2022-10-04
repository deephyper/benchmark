import importlib
import importlib.util
import logging
import os
import sys

PKG_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BCH_ROOT_DIR = os.path.join(os.path.dirname(PKG_ROOT_DIR), "lib")

def find_benchmark(benchmark_name):

    benchmark_path = os.path.join(BCH_ROOT_DIR, benchmark_name)
    sys.path.insert(0, benchmark_path)
    benchmark_module = importlib.import_module("benchmark")
   
    benchmark_class = None
    for attr_name in benchmark_module.__dict__:
        attr = getattr(benchmark_module, attr_name)
        try:
            if issubclass(attr, Benchmark) and not(attr is Benchmark):
                benchmark_class = attr
                break
        except TypeError: pass
    
    if benchmark_class is None:
        logging.error("cannot find any benchmark")
        return
    else:
        logging.info(f"found {attr.__name__}")
    
    benchmark = benchmark_class()
    benchmark.cwd = benchmark_path
    benchmark.name = benchmark_name.replace("-","_").replace("/",".").lower()

    return benchmark

def install(name):
    logging.info(f"installing {name}") 
    
    benchmark = find_benchmark(name)
    benchmark.install()

    logging.info(f"installation done")
    

def load(name):
    logging.info(f"loading {name}")

    benchmark = find_benchmark(name)
    module = benchmark.load()

    logging.info(f"loading done")

    return module




class Benchmark:

    def __init__(self):
        self.name = None
        self.cwd = None

    def install(self):
        
        for require_name, require_val in self.requires.items():
            res = 0
            if require_val["type"] == "cmd":
                logging.info(f"running requires[{require_name}]")
                cmd = require_val["cmd"]
                res = os.system(f"cd {self.cwd} && {cmd}")
            elif require_val["type"] == "pip":
                name = require_val["name"]
                res = os.system(f"pip install {name}")

            if res != 0:
                logging.error("installation failed!")
    
    def load(self):
        for require_name, require_val in self.requires.items():
            if require_val["type"] == "pythonpath":
                path = require_val["path"]
                logging.info(f"adding {path} to PYTHONPATH")
                sys.path.insert(0, path)

        module_name = f"deephyper_benchmark.lib.{self.name}"
        spec = importlib.util.spec_from_file_location(module_name, f"{self.cwd}/__init__.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self.module = module
    
        return self.module
