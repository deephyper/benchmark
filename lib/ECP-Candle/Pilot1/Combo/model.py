from __future__ import division, print_function

import collections
import logging
import os
import time
import warnings

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "combo_default_model.yaml")

#! must be placed before "import candle"
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateScheduler,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_custom_objects

import candle
import combo
import NCI60
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from deephyper_benchmark.integration.keras import count_params
from deephyper_benchmark.utils.json_utils import array_to_json


logger = logging.getLogger(__name__)

np.set_printoptions(precision=4)
tf.compat.v1.disable_eager_execution()


def verify_path(path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def extension_from_parameters(args):
    """Construct string for saving model with annotation of parameters"""
    ext = ""
    ext += ".A={}".format(args.activation)
    ext += ".B={}".format(args.batch_size)
    ext += ".E={}".format(args.epochs)
    ext += ".O={}".format(args.optimizer)
    # ext += '.LEN={}'.format(args.maxlen)
    ext += ".LR={}".format(args.learning_rate)
    ext += ".CF={}".format("".join([x[0] for x in sorted(args.cell_features)]))
    ext += ".DF={}".format("".join([x[0] for x in sorted(args.drug_features)]))
    if args.feature_subsample > 0:
        ext += ".FS={}".format(args.feature_subsample)
    if args.dropout > 0:
        ext += ".DR={}".format(args.dropout)
    if args.warmup_lr:
        ext += ".wu_lr"
    if args.reduce_lr:
        ext += ".re_lr"
    if args.residual:
        ext += ".res"
    if args.use_landmark_genes:
        ext += ".L1000"
    if args.gen:
        ext += ".gen"
    if args.use_combo_score:
        ext += ".scr"
    if args.use_mean_growth:
        ext += ".mg"
    for i, n in enumerate(args.dense):
        if n > 0:
            ext += ".D{}={}".format(i + 1, n)
    if args.dense_feature_layers != args.dense:
        for i, n in enumerate(args.dense):
            if n > 0:
                ext += ".FD{}={}".format(i + 1, n)

    return ext


def discretize(y, bins=5):
    percentiles = [100 / bins * (i + 1) for i in range(bins - 1)]
    thresholds = [np.percentile(y, x) for x in percentiles]
    classes = np.digitize(y, thresholds)
    return classes


class ComboDataLoader(object):
    """Load merged drug response, drug descriptors and cell line essay data"""

    def __init__(
        self,
        seed,
        valid_split=0.2,
        test_split=0.2,
        shuffle=True,
        cell_features=["expression"],
        drug_features=["descriptors"],
        response_url=None,
        use_landmark_genes=False,
        use_combo_score=False,
        use_mean_growth=False,
        preprocess_rnaseq=None,
        exclude_cells=[],
        exclude_drugs=[],
        feature_subsample=None,
        scaling="std",
        scramble=False,
    ):
        """Initialize data merging drug response, drug descriptors and cell line essay.
           Shuffle and split training and validation set

        Parameters
        ----------
        seed: integer
            seed for random generation
        val_split : float, optional (default 0.2)
            fraction of data to use in validation
        cell_features: list of strings from 'expression', 'expression_5platform', 'mirna', 'proteome', 'all', 'categorical' (default ['expression'])
            use one or more cell line feature sets: gene expression, microRNA, proteome
            use 'all' for ['expression', 'mirna', 'proteome']
            use 'categorical' for one-hot encoded cell lines
        drug_features: list of strings from 'descriptors', 'latent', 'all', 'categorical', 'noise' (default ['descriptors'])
            use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder
            trained on NSC drugs, or both; use random features if set to noise
            use 'categorical' for one-hot encoded drugs
        shuffle : True or False, optional (default True)
            if True shuffles the merged data before splitting training and validation sets
        scramble: True or False, optional (default False)
            if True randomly shuffle dose response data as a control
        feature_subsample: None or integer (default None)
            number of feature columns to use from cellline expressions and drug descriptors
        use_landmark_genes: True or False
            only use LINCS1000 landmark genes
        use_combo_score: bool (default False)
            use combination score in place of percent growth (stored in 'GROWTH' column)
        use_mean_growth: bool (default False)
            use mean aggregation instead of min on percent growth
        scaling: None, 'std', 'minmax' or 'maxabs' (default 'std')
            type of feature scaling: 'maxabs' to [-1,1], 'maxabs' to [-1, 1], 'std' for standard normalization
        """

        self._random_state = np.random.RandomState(seed)

        df = NCI60.load_combo_response(
            response_url=response_url,
            use_combo_score=use_combo_score,
            use_mean_growth=use_mean_growth,
            fraction=True,
            exclude_cells=exclude_cells,
            exclude_drugs=exclude_drugs,
        )
        logger.info("Loaded {} unique (CL, D1, D2) response sets.".format(df.shape[0]))

        if "all" in cell_features:
            self.cell_features = ["expression", "mirna", "proteome"]
        else:
            self.cell_features = cell_features

        if "all" in drug_features:
            self.drug_features = ["descriptors", "latent"]
        else:
            self.drug_features = drug_features

        for fea in self.cell_features:
            if fea == "expression" or fea == "rnaseq":
                self.df_cell_expr = NCI60.load_cell_expression_rnaseq(
                    ncols=feature_subsample,
                    scaling=scaling,
                    use_landmark_genes=use_landmark_genes,
                    preprocess_rnaseq=preprocess_rnaseq,
                )
                df = df.merge(self.df_cell_expr[["CELLNAME"]], on="CELLNAME")
            elif fea == "expression_u133p2":
                self.df_cell_expr = NCI60.load_cell_expression_u133p2(
                    ncols=feature_subsample,
                    scaling=scaling,
                    use_landmark_genes=use_landmark_genes,
                )
                df = df.merge(self.df_cell_expr[["CELLNAME"]], on="CELLNAME")
            elif fea == "expression_5platform":
                self.df_cell_expr = NCI60.load_cell_expression_5platform(
                    ncols=feature_subsample,
                    scaling=scaling,
                    use_landmark_genes=use_landmark_genes,
                )
                df = df.merge(self.df_cell_expr[["CELLNAME"]], on="CELLNAME")
            elif fea == "mirna":
                self.df_cell_mirna = NCI60.load_cell_mirna(
                    ncols=feature_subsample, scaling=scaling
                )
                df = df.merge(self.df_cell_mirna[["CELLNAME"]], on="CELLNAME")
            elif fea == "proteome":
                self.df_cell_prot = NCI60.load_cell_proteome(
                    ncols=feature_subsample, scaling=scaling
                )
                df = df.merge(self.df_cell_prot[["CELLNAME"]], on="CELLNAME")
            elif fea == "categorical":
                df_cell_ids = df[["CELLNAME"]].drop_duplicates()
                cell_ids = df_cell_ids["CELLNAME"].map(lambda x: x.replace(":", "."))
                df_cell_cat = pd.get_dummies(cell_ids)
                df_cell_cat.index = df_cell_ids["CELLNAME"]
                self.df_cell_cat = df_cell_cat.reset_index()

        for fea in self.drug_features:
            if fea == "descriptors":
                self.df_drug_desc = NCI60.load_drug_descriptors(
                    ncols=feature_subsample, scaling=scaling
                )
                df = df[
                    df["NSC1"].isin(self.df_drug_desc["NSC"])
                    & df["NSC2"].isin(self.df_drug_desc["NSC"])
                ]
            elif fea == "latent":
                self.df_drug_auen = NCI60.load_drug_autoencoded_AG(
                    ncols=feature_subsample, scaling=scaling
                )
                df = df[
                    df["NSC1"].isin(self.df_drug_auen["NSC"])
                    & df["NSC2"].isin(self.df_drug_auen["NSC"])
                ]
            elif fea == "categorical":
                df_drug_ids = df[["NSC1"]].drop_duplicates()
                df_drug_ids.columns = ["NSC"]
                drug_ids = df_drug_ids["NSC"]
                df_drug_cat = pd.get_dummies(drug_ids)
                df_drug_cat.index = df_drug_ids["NSC"]
                self.df_drug_cat = df_drug_cat.reset_index()
            elif fea == "noise":
                ids1 = df[["NSC1"]].drop_duplicates().rename(columns={"NSC1": "NSC"})
                ids2 = df[["NSC2"]].drop_duplicates().rename(columns={"NSC2": "NSC"})
                df_drug_ids = pd.concat([ids1, ids2]).drop_duplicates()
                noise = np.random.normal(size=(df_drug_ids.shape[0], 500))
                df_rand = pd.DataFrame(
                    noise,
                    index=df_drug_ids["NSC"],
                    columns=["RAND-{:03d}".format(x) for x in range(500)],
                )
                self.df_drug_rand = df_rand.reset_index()

        logger.info(
            "Filtered down to {} rows with matching information.".format(df.shape[0])
        )

        ids1 = df[["NSC1"]].drop_duplicates().rename(columns={"NSC1": "NSC"})
        ids2 = df[["NSC2"]].drop_duplicates().rename(columns={"NSC2": "NSC"})
        df_drug_ids = pd.concat([ids1, ids2]).drop_duplicates().reset_index(drop=True)

        n_drugs = df_drug_ids.shape[0]
        n_valid_drugs = int(n_drugs * valid_split)
        n_test_drugs = int(n_drugs * test_split)
        n_train_drugs = n_drugs - n_valid_drugs - n_test_drugs

        logger.info("Unique cell lines: {}".format(df["CELLNAME"].nunique()))
        logger.info("Unique drugs: {}".format(n_drugs))

        if shuffle:
            df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            df_drug_ids = df_drug_ids.sample(frac=1.0, random_state=seed).reset_index(
                drop=True
            )

        self.df_response = df
        self.df_drug_ids = df_drug_ids

        self.train_drug_ids = df_drug_ids["NSC"][:n_train_drugs]
        self.valid_drug_ids = df_drug_ids["NSC"][
            n_train_drugs : n_train_drugs + n_valid_drugs
        ]
        self.test_drug_ids = df_drug_ids["NSC"][
            n_train_drugs + n_valid_drugs : n_train_drugs + n_valid_drugs + n_test_drugs
        ]

        if scramble:
            growth = df[["GROWTH"]]
            random_growth = growth.iloc[
                self._random_state.permutation(np.arange(growth.shape[0]))
            ].reset_index()
            self.df_response[["GROWTH"]] = random_growth["GROWTH"]
            logger.warn("Randomly shuffled dose response growth values.")

        logger.info("Distribution of dose response:")
        logger.info(self.df_response[["GROWTH"]].describe())

        self.total = df.shape[0]
        self.n_valid = int(self.total * valid_split)
        self.n_test = int(self.total * test_split)
        self.n_train = self.total - self.n_valid - self.n_test
        logger.info(
            "Rows in train: {}, valid: {}, test: {}".format(
                self.n_train, self.n_valid, self.n_test
            )
        )

        self.cell_df_dict = {
            "expression": "df_cell_expr",
            "expression_5platform": "df_cell_expr",
            "expression_u133p2": "df_cell_expr",
            "rnaseq": "df_cell_expr",
            "mirna": "df_cell_mirna",
            "proteome": "df_cell_prot",
            "categorical": "df_cell_cat",
        }

        self.drug_df_dict = {
            "descriptors": "df_drug_desc",
            "latent": "df_drug_auen",
            "categorical": "df_drug_cat",
            "noise": "df_drug_rand",
        }

        self.input_features = collections.OrderedDict()
        self.feature_shapes = {}
        for fea in self.cell_features:
            feature_type = "cell." + fea
            feature_name = "cell." + fea
            df_cell = getattr(self, self.cell_df_dict[fea])
            self.input_features[feature_name] = feature_type
            self.feature_shapes[feature_type] = (df_cell.shape[1] - 1,)

        for drug in ["drug1", "drug2"]:
            for fea in self.drug_features:
                feature_type = "drug." + fea
                feature_name = drug + "." + fea
                df_drug = getattr(self, self.drug_df_dict[fea])
                self.input_features[feature_name] = feature_type
                self.feature_shapes[feature_type] = (df_drug.shape[1] - 1,)

        logger.info("Input features shapes:")
        for k, v in self.input_features.items():
            logger.info("  {}: {}".format(k, self.feature_shapes[v]))

        self.input_dim = sum(
            [np.prod(self.feature_shapes[x]) for x in self.input_features.values()]
        )
        logger.info("Total input dimensions: {}".format(self.input_dim))

    def load_data_all(self, switch_drugs=False):
        df_all = self.df_response
        y_all = df_all["GROWTH"].values
        x_all_list = []

        for fea in self.cell_features:
            df_cell = getattr(self, self.cell_df_dict[fea])
            df_x_all = pd.merge(
                df_all[["CELLNAME"]], df_cell, on="CELLNAME", how="left"
            )
            x_all_list.append(df_x_all.drop(["CELLNAME"], axis=1).values)

        drugs = ["NSC1", "NSC2"]
        if switch_drugs:
            drugs = ["NSC2", "NSC1"]

        for drug in drugs:
            for fea in self.drug_features:
                df_drug = getattr(self, self.drug_df_dict[fea])
                df_x_all = pd.merge(
                    df_all[[drug]], df_drug, left_on=drug, right_on="NSC", how="left"
                )
                x_all_list.append(df_x_all.drop([drug, "NSC"], axis=1).values)

        return x_all_list, y_all, df_all

    def load_data_by_index(self, train_index, valid_index, test_index):
        x_all_list, y_all, df_all = self.load_data_all()

        x_train_list = [x[train_index] for x in x_all_list]
        x_valid_list = [x[valid_index] for x in x_all_list]
        x_test_list = [x[test_index] for x in x_all_list]

        y_train = y_all[train_index]
        y_valid = y_all[valid_index]
        y_test = y_all[test_index]

        df_train = df_all.iloc[train_index, :]
        df_valid = df_all.iloc[valid_index, :]
        df_test = df_all.iloc[test_index, :]

        return (
            x_train_list,
            y_train,
            x_valid_list,
            y_valid,
            x_test_list,
            y_test,
            df_train,
            df_valid,
            df_test,
        )

    def load_data(self):
        train_index = range(self.n_train)
        valid_index = range(self.n_train, self.n_train + self.n_valid)
        test_index = range(
            self.n_train + self.n_valid, self.n_train + self.n_valid + self.n_test
        )
        return self.load_data_by_index(train_index, valid_index, test_index)


def test_loader(loader):
    (
        x_train_list,
        y_train,
        x_val_list,
        y_val,
        x_test_list,
        y_test,
        _,
        _,
        _,
    ) = loader.load_data()
    print("x_train shapes:")
    for x in x_train_list:
        print(x.shape)
    print("y_train shape:", y_train.shape)

    print("x_valid shapes:")
    for x in x_val_list:
        print(x.shape)
    print("y_valid shape:", y_val.shape)

    print("x_test shapes:")
    for x in x_test_list:
        print(x.shape)
    print("y_test shape:", y_test.shape)


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def mae(y_true, y_pred):
    return keras.metrics.mean_absolute_error(y_true, y_pred)


def mse(y_true, y_pred):
    return keras.metrics.mean_squared_error(y_true, y_pred)


def evaluate_prediction(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    return {"mse": mse, "mae": mae, "r2": r2, "corr": corr}


def log_evaluation(metric_outputs, description="Comparing y_true and y_pred:"):
    logger.info(description)
    for metric, value in metric_outputs.items():
        logger.info("  {}: {:.4f}".format(metric, value))


class LoggingCallback(Callback):
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "[Epoch: %i] %s" % (
            epoch,
            ", ".join("%s: %f" % (k, v) for k, v in sorted(logs.items())),
        )
        self.print_fcn(msg)


class PermanentDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(PermanentDropout, self).__init__(rate, **kwargs)
        self.uses_learning_phase = False

    def call(self, x, mask=None):
        if 0.0 < self.rate < 1.0:
            noise_shape = self._get_noise_shape(x)
            x = K.dropout(x, self.rate, noise_shape)
        return x


class ModelRecorder(Callback):
    def __init__(self, save_all_models=False):
        Callback.__init__(self)
        self.save_all_models = save_all_models
        get_custom_objects()["PermanentDropout"] = PermanentDropout

    def on_train_begin(self, logs={}):
        self.val_losses = []
        self.best_val_loss = np.Inf
        self.best_model = None

    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs.get("val_loss")
        self.val_losses.append(val_loss)
        if val_loss < self.best_val_loss:
            self.best_model = keras.models.clone_model(self.model)
            self.best_val_loss = val_loss


def build_feature_model(
    input_shape,
    name="",
    dense_layers=[1000, 1000],
    activation="relu",
    residual=False,
    dropout_rate=0,
    permanent_dropout=True,
):
    x_input = Input(shape=input_shape)
    h = x_input
    for i, layer in enumerate(dense_layers):
        x = h
        h = Dense(layer, activation=activation)(h)
        if dropout_rate > 0:
            if permanent_dropout:
                h = PermanentDropout(dropout_rate)(h)
            else:
                h = Dropout(dropout_rate)(h)
        if residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    model = Model(x_input, h, name=name)
    return model


def build_model(loader, args, verbose=False):
    input_models = {}
    dropout_rate = args.dropout
    permanent_dropout = True
    for fea_type, shape in loader.feature_shapes.items():
        box = build_feature_model(
            input_shape=shape,
            name=fea_type,
            dense_layers=args.dense_feature_layers,
            dropout_rate=dropout_rate,
            permanent_dropout=permanent_dropout,
        )
        if verbose:
            box.summary()
        input_models[fea_type] = box

    inputs = []
    encoded_inputs = []
    for fea_name, fea_type in loader.input_features.items():
        shape = loader.feature_shapes[fea_type]
        fea_input = Input(shape, name="input." + fea_name)
        inputs.append(fea_input)
        input_model = input_models[fea_type]
        encoded = input_model(fea_input)
        encoded_inputs.append(encoded)

    merged = keras.layers.concatenate(encoded_inputs)

    h = merged
    for i, layer in enumerate(args.dense):
        x = h
        h = Dense(layer, activation=args.activation)(h)
        if dropout_rate > 0:
            if permanent_dropout:
                h = PermanentDropout(dropout_rate)(h)
            else:
                h = Dropout(dropout_rate)(h)
        if args.residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    output = Dense(1)(h)

    return Model(inputs, output)


def initialize_parameters(default_model="combo_default_model.txt"):
    # Build benchmark object
    comboBmk = combo.BenchmarkCombo(
        combo.file_path,
        default_model,
        "keras",
        prog="combo_baseline",
        desc="Build neural network based models to predict tumor response to drug pairs.",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(comboBmk)

    return gParameters


def yaml_load(path):
    with open(path, "r") as f:
        yaml_data = yaml.load(f, Loader=Loader)
    return yaml_data


def run_pipeline(config: dict = None, mode="valid", stopper_callback=None):
    # Default Config from original Benchmark
    params = initialize_parameters()

    # Default Config from our Benchmark
    params.update(yaml_load(DEFAULT_CONFIG))
    params.pop("val_split")

    # Ingest input configuration
    if config:
        params.update(config)

    args = candle.ArgumentStruct(**params)
    seed = args.rng_seed
    candle.set_seed(seed)

    loader = ComboDataLoader(
        seed=args.rng_seed,
        valid_split=args.valid_split,
        test_split=args.test_split,
        cell_features=args.cell_features,
        drug_features=args.drug_features,
        use_mean_growth=args.use_mean_growth,
        response_url=args.response_url,
        use_landmark_genes=args.use_landmark_genes,
        preprocess_rnaseq=args.preprocess_rnaseq,
        exclude_cells=args.exclude_cells,
        exclude_drugs=args.exclude_drugs,
        use_combo_score=args.use_combo_score,
        scaling=args.scaling,
    )

    # test_loader(loader)

    model = build_model(loader, args, verbose=args.verbose)

    def warmup_scheduler(epoch):
        lr = args.learning_rate or base_lr * args.batch_size / 100
        if epoch <= 5:
            K.set_value(model.optimizer.lr, (base_lr * (5 - epoch) + lr * epoch) / 5)
        logger.debug("Epoch {}: lr={}".format(epoch, K.get_value(model.optimizer.lr)))
        return K.get_value(model.optimizer.lr)

    model = build_model(loader, args)

    optimizer = optimizers.deserialize({"class_name": args.optimizer, "config": {}})
    base_lr = args.base_lr or K.get_value(optimizer.lr)
    if args.learning_rate:
        K.set_value(optimizer.lr, args.learning_rate)

    model.compile(loss=args.loss, optimizer=optimizer, metrics=[mse, mae, r2])

    # calculate trainable and non-trainable params
    params.update(candle.compute_trainable_params(model))

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=args.early_stopping_patience
    )

    timeout_monitor = candle.TerminateOnTimeOut(params["timeout"])

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=args.reduce_lr_factor,
        patience=args.reduce_lr_patience,
        min_lr=0.00001,
    )
    warmup_lr = LearningRateScheduler(warmup_scheduler)

    callbacks = [timeout_monitor]
    if args.early_stopping:
        callbacks.append(early_stopping)
    if args.reduce_lr:
        callbacks.append(reduce_lr)
    if args.warmup_lr:
        callbacks.append(warmup_lr)

    if stopper_callback:
        callbacks.append(stopper_callback)

    (
        x_train_list,
        y_train,
        x_valid_list,
        y_valid,
        x_test_list,
        y_test,
        _,
        _,
        _,
    ) = loader.load_data()
    training_data = (x_train_list, y_train)
    validation_data = (x_valid_list, y_valid)

    num_parameters_info = count_params(model)

    # Training
    timestamp_duration = time.time() 
    history = model.fit(
        *training_data,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        epochs=args.epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=args.verbose,
    ).history
    duration_train = time.time() - timestamp_duration


    timestamp_duration = time.time() 
    y_train_pred = model.predict(x_train_list, batch_size=args.batch_size).flatten()
    duration_batch_inference = (time.time() - timestamp_duration) / np.ceil(len(y_train_pred) / args.batch_size)

    scores_train = evaluate_prediction(y_train, y_train_pred)
    y_valid_pred = model.predict(x_valid_list, batch_size=args.batch_size).flatten()
    scores_valid = evaluate_prediction(y_valid, y_valid_pred)
    y_test_pred = model.predict(x_test_list, batch_size=args.batch_size).flatten()
    scores_test = evaluate_prediction(y_test, y_test_pred)

    all_scores = {}
    all_scores.update({f"train_{k}": float(v) for k, v in scores_train.items()})
    all_scores.update({f"valid_{k}": float(v) for k, v in scores_valid.items()})
    all_scores.update({f"test_{k}": float(v) for k, v in scores_test.items()})

    if K.backend() == "tensorflow":
        K.clear_session()

    objective = all_scores["valid_r2"]  # validation R2
    objective = max(-1, objective)

    # collect learning curves
    lc_train_mse = array_to_json(
        np.asarray([[i + 1, l] for i, l in enumerate(history["mse"])])
    )
    lc_valid_mse = array_to_json(
        np.asarray([[i + 1, l] for i, l in enumerate(history["val_mse"])])
    )

    lc_train_mae = array_to_json(
        np.asarray([[i + 1, l] for i, l in enumerate(history["mae"])])
    )
    lc_valid_mae = array_to_json(
        np.asarray([[i + 1, l] for i, l in enumerate(history["val_mae"])])
    )

    lc_train_r2 = array_to_json(
        np.asarray([[i + 1, l] for i, l in enumerate(history["r2"])])
    )
    lc_valid_r2 = array_to_json(
        np.asarray([[i + 1, l] for i, l in enumerate(history["val_r2"])])
    )

    metadata = {
        "num_parameters": num_parameters_info["num_parameters"],
        "num_parameters_train": num_parameters_info["num_parameters_train"],
        "duration_train": duration_train,
        "duration_batch_inference": duration_batch_inference,
        "budget": len(history["loss"]),
        "stopped": len(history["loss"]) < args.epochs,
        "lc_train_mse": lc_train_mse,
        "lc_valid_mse": lc_valid_mse,
        "lc_train_mae": lc_train_mae,
        "lc_valid_mae": lc_valid_mae,
        "lc_train_r2": lc_train_r2,
        "lc_valid_r2": lc_valid_r2,
    }
    metadata.update(all_scores)
    return {"objective": objective, "metadata": metadata}
