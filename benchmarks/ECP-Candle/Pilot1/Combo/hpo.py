import traceback

from deephyper.evaluator import profile
from deephyper.problem import HpProblem


problem = HpProblem()

# Model hyperparameters
ACTIVATIONS = [
    "elu",
    "gelu",
    "hard_sigmoid",
    "linear",
    "relu",
    "selu",
    "sigmoid",
    "softplus",
    "softsign",
    "swish",
    "tanh",
]
default_dense = [1000, 1000, 1000]
default_dense_feature_layers = [1000, 1000, 1000]

for i in range(len(default_dense)):

    problem.add_hyperparameter(
        (10, 1024, "log-uniform"),
        f"dense_{i}",
        default_value=default_dense[i],
    )

    problem.add_hyperparameter(
        (10, 1024, "log-uniform"),
        f"dense_feature_layers_{i}",
        default_value=default_dense_feature_layers[i],
    )

problem.add_hyperparameter(ACTIVATIONS, "activation", default_value="relu")

# Optimization hyperparameters
problem.add_hyperparameter(
    [
        "sgd",
        "rmsprop",
        "adagrad",
        "adadelta",
        "adam",
    ],
    "optimizer",
    default_value="sgd",
)

problem.add_hyperparameter((0, 0.5), "dropout", default_value=0.0)
problem.add_hyperparameter((8, 512, "log-uniform"), "batch_size", default_value=32)

problem.add_hyperparameter(
    (1e-5, 1e-2, "log-uniform"), "learning_rate", default_value=0.001
)
problem.add_hyperparameter((1e-5, 1e-2, "log-uniform"), "base_lr", default_value=0.001)
problem.add_hyperparameter([True, False], "residual", default_value=False)

problem.add_hyperparameter([True, False], "early_stopping", default_value=False)
problem.add_hyperparameter((5, 20), "early_stopping_patience", default_value=5)

problem.add_hyperparameter([True, False], "reduce_lr", default_value=False)
problem.add_hyperparameter((0.1, 1.0), "reduce_lr_factor", default_value=0.5)
problem.add_hyperparameter((5, 20), "reduce_lr_patience", default_value=5)

problem.add_hyperparameter([True, False], "warmup_lr", default_value=False)
problem.add_hyperparameter([True, False], "batch_normalization", default_value=False)

problem.add_hyperparameter(
    ["mse", "mae", "logcosh", "mape", "msle", "huber"], "loss", default_value="mse"
)

problem.add_hyperparameter(["std", "minmax", "maxabs"], "scaling", default_value="std")


@profile
def run(config, verbose=0):
    params = initialize_parameters()

    params["epochs"] = 100
    # params["timeout"] = 60 * 20  # 20 minutes per model
    params["cp"] = False
    params["verbose"] = verbose
    params["tb"] = False

    if len(config) > 0:
        dense = []
        dense_feature_layers = []
        for i in range(len(default_dense)):

            key = f"dense_{i}"
            dense.append(config.pop(key))

            key = f"dense_feature_layers_{i}"
            dense_feature_layers.append(config.pop(key))

        config["dense"] = dense
        config["dense_feature_layers"] = dense_feature_layers
        params.update(config)

    try:
        score = run_candle(params)

        if isinstance(score, dict):
            score["objective"] = max(score["objective"], -1)
        else:
            score = max(score, -1)

    except Exception as e:
        print(traceback.format_exc())
        score = -1

        if "optuna_trial" in params:
            score = {"objective": score, "step": 1, "pruned": True}

    return score
