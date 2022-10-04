import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
    force=True,
)

from deephyper_benchmark import *

# install("ECP-Candle/Pilot1/Combo")

load("ECP-Candle/Pilot1/Combo")

# required before importing candle
from tensorflow.keras import backend as K

import candle
import combo