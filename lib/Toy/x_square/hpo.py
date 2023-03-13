from deephyper.problem import HpProblem

problem = HpProblem()
problem.add_hyperparameter((-10.0, 10.0), "x")


def run(job):
    return -job.parameters["x"] ** 2
