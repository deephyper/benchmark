# C-BBO: Continuous Black-Box Optimization

> **Warning**
> Work in progress, this benchmark is not yet ready.

Set of continuous function benchmarks. The `DEEPHYPER_BENCHMARK_NDIMS` environment variable defines the number of dimensions of the problem.

| Function Name | Number of Dimensions  |                   Comment                   |
| ------------- | --------------------- | ------------------------------------------- |
| ackley        | $\infty$ (default 5)  | Many local minima and single global optimum |
| branin        | 2                     | Three global optimum                        |
| cossin        | 1                     | Many local minima, good for visualisation.  |
| easom         | 2                     | Almost flat everywhere                      |
| griewank      | $\infty$ (default 5)  |                                             |
| hartmann6D    | 6                     |                                             |
| levy          | $\infty$ (default 5)  |                                             |
| michal        | $\infty$ (default 2)  |                                             |
| rosen         | $\infty$ (default 5)  |                                             |
| schwefel      | $\infty$ (default 5)  |                                             |
| shekel        | 4                     | Many local minima with flat areas           |