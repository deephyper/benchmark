import os

from deephyper_benchmark import *

DIR = os.path.dirname(os.path.abspath(__file__))


class YAHPOLCBench(Benchmark):
    version = "0.0.1"

    data_dir = os.path.join(DIR, "..", "build", "data")
    requires = {
        "download-data": {
            "step": "install",
            "type": "cmd",
            "cmd": f"git clone https://github.com/slds-lmu/yahpo_data.git {data_dir}",
        },
        "pip-yahpo-gym": {"step": "install", "type": "pip", "args": "install yahpo-gym"},
    }

    def install(self):
        super().install()

        from yahpo_gym import local_config

        local_config.init_config()
        local_config.set_data_path(self.data_dir)
