import numpy as np

import lcdb

from deephyper.evaluator import profile, RunningJob
from deephyper.hpo import HpProblem


df_meta = lcdb.get_meta_features()

all_openmlid = df_meta["openmlid"].tolist()
all_name = df_meta["Name"].tolist()
all_algorithms = [
    "SVC_linear",
    "SVC_poly",
    "SVC_rbf",
    "SVC_sigmoid",
    "sklearn.tree.DecisionTreeClassifier",
    "sklearn.tree.ExtraTreeClassifier",
    "sklearn.linear_model.LogisticRegression",
    "sklearn.linear_model.PassiveAggressiveClassifier",
    "sklearn.linear_model.Perceptron",
    "sklearn.linear_model.RidgeClassifier",
    "sklearn.linear_model.SGDClassifier",
    "sklearn.neural_network.MLPClassifier",
    # "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
    # "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis",
    "sklearn.naive_bayes.BernoulliNB",
    # "sklearn.naive_bayes.MultinomialNB", # often missing for large datasets
    "sklearn.neighbors.KNeighborsClassifier",
    "sklearn.ensemble.ExtraTreesClassifier",
    "sklearn.ensemble.RandomForestClassifier",
    "sklearn.ensemble.GradientBoostingClassifier",
]

problem = HpProblem()
problem.add_hyperparameter(all_algorithms, "model")


@profile
def run(job: RunningJob, optuna_trial=None, task_id=3) -> dict:

    model = job.parameters["model"]

    try:
        curve = lcdb.get_curve(task_id, model, metric="accuracy")
    except Exception:
        if optuna_trial:
            objective = 0  # accuracy 0
        else:
            objective = [[1], [0]]  # accuracy 0 at step 1
        return {
            "objective": objective,
            "metadata": {
                "budget": budget_i,
                "stopped": budget_i < anchors[-1],
                "duration": cum_time,
            },
        }

    anchors, _, scores_valid, _ = curve
    _, times = lcdb.get_train_times(task_id, model)

    # scores_valid = [list(v) for v in scores_valid]
    # times = [list(t) for t in times]

    anchors = np.array(anchors).tolist()
    # print(model, type(scores_valid), type(scores_valid[0]), type(scores_valid[0][0]))
    # scores_valid = np.array(np.array(scores_valid).tolist())
    # times = np.array(np.array(times).tolist())

    # print("->", np.ndim(scores_valid))
    # print("->", scores_valid)
    # print("->", np.shape(scores_valid))

    # if np.ndim(scores_valid) == 2:
    # scores_valid = np.mean(scores_valid, axis=1)
    scores_valid = np.array([v[0] for v in scores_valid])
    times = np.array([t[0] for t in times])

    scores_valid = scores_valid.tolist()
    times = times.tolist()

    cum_time = 0

    if optuna_trial:

        for i, budget_i in enumerate(anchors):
            objective_i = scores_valid[i]
            cum_time += times[i]
            optuna_trial.report(objective_i, step=budget_i)
            if optuna_trial.should_prune():
                break

        return {
            "objective": objective_i,
            "metadata": {
                "budget": budget_i,
                "stopped": budget_i < anchors[-1],
                "duration": cum_time,
            },
        }

    else:

        for i, budget_i in enumerate(anchors):
            objective_i = scores_valid[i]
            cum_time += times[i]
            job.record(budget_i, objective_i)
            if job.stopped():
                break

        return {
            "objective": job.observations,
            "metadata": {
                "budget": budget_i,
                "stopped": budget_i < anchors[-1],
                "duration": cum_time,
            },
        }


if __name__ == "__main__":
    print(problem)
    default_config = problem.default_configuration
    print(f"{default_config=}")
    result = run(RunningJob(parameters=default_config))
    print(f"{result=}")
