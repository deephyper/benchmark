import os

DIR = os.path.dirname(os.path.abspath(__file__))

from hpobench.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark

b = SliceLocalizationBenchmark(
    data_path=os.path.join(DIR, "../build/HPOBench/data/fcnet_tabular_benchmarks/")
)
print(b.fidelity_space)

config = b.get_configuration_space(seed=1).sample_configuration()

result_dict = b.objective_function(configuration=config, fidelity={"budget": 50}, rng=1)

# returns results on the highest budget
result_dict = b.objective_function(configuration=config, rng=1)

print(result_dict)
