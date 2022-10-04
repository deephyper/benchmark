
```python
from deephyper_benchmark import *

install("ECP-Candle/Pilot1/Combo")

load("ECP-Candle/Pilot1/Combo")

# required before importing candle
from tensorflow.keras import backend as K

import candle
import combo
```