from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark

def run_test(config):
    result_dict = NNBenchmark(task_id=1).objective_function(configuration=config, fidelity={"n_estimators": 128, "dataset_fraction": 0.5}, rng=1)
    print(f"HAYO: {result_dict.keys()}")
    return 1 #result_dict["idontknowthekey"]