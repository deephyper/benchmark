import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
    force=True,
)

import deephyper_benchmark as dhb

dhb.install("ECP-Candle/Pilot1/Combo")

dhb.load("ECP-Candle/Pilot1/Combo")

# Run training with Testing scores
from deephyper_benchmark.lib.ecp_candle.pilot1.combo import model 
res = model.run_pipeline(mode="test")
print(f"{res=}")

# Run HPO-pipeline with default configuration of hyperparameters
from deephyper_benchmark.lib.ecp_candle.pilot1.combo import hpo
config = hpo.problem.default_configuration
res = hpo.run(config)
print(f"{res=}")


