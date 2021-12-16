import random
import time

def run_sleep(config):
    if config["random_duration"]:
        time.sleep(random.uniform(0, config["run_duration"]))
    else:
        time.sleep(config["run_duration"])
    return 0