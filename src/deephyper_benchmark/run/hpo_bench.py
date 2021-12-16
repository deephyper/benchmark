from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark

def run_test(config):
    if config.get("id"):
        del(config["id"])
    result_dict = NNBenchmark(task_id=1).objective_function(configuration=config, rng=1)
    print(f"result_dict keys : {result_dict.keys()}")
    return 1 #result_dict["idontknowthekey"]