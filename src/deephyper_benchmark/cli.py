import argparse
# import importlib.util
import importlib
import logging
import os
import sys
import pathlib
import json

import matplotlib.pyplot as plt

from deephyper_benchmark.benchmark import Benchmark

logger = logging.getLogger(__name__)


def _create_parser():
    parser = argparse.ArgumentParser(description="DeepHyper/Benchmark command line.")
    parser.add_argument("script", type=str)
    parser.add_argument("-o", "--output", type=str, required=False, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _write_report(output, report):

    if output is not None and os.path.exists(output):
        raise ValueError("The 'output' target already exist!")

    err_msg = "'{}' is not found the the {}"
    assert "num_workers" in report, err_msg.format("num_workers", "report")
    assert "profile" in report, err_msg.format("profile", "report")
    assert "search" in report, err_msg.format("search", "report")

    num_workers = report["num_workers"]
    profile = report["profile"]
    search = report["search"]

    # keys of profile: timestamp n_jobs_running
    assert "timestamp" in profile.columns, err_msg.format("timestamp", "profile")
    assert "n_jobs_running" in profile.columns, err_msg.format(
        "n_jobs_running", "profile"
    )

    # compute worker utilization
    t0 = profile.iloc[0].timestamp
    t_max = profile.iloc[-1].timestamp
    T_max = (t_max - t0) * num_workers

    cum = 0
    for i in range(len(profile.timestamp)-1):
        cum += (
            profile.timestamp.iloc[i + 1] - profile.timestamp.iloc[i]
        ) * profile.n_jobs_running.iloc[i]
    perc_util = cum / T_max

    if output is not None:

        pathlib.Path(output).mkdir(parents=True, exist_ok=False)

        # saving report
        profile.to_csv(os.path.join(output, "profile.csv"))
        search.to_csv(os.path.join(output, "search.csv"))

        infos = {
            "num_workers": num_workers,
            "duration": T_max,
            "perc_util": perc_util
            }
        with open(os.path.join(output, "infos.json"), "w") as fp:
            json.dump(infos, fp, indent=2)

        # 1st figure: number of jobs vs time
        plt.figure()
        plt.step(profile.timestamp - t0, profile.n_jobs_running, where="post")
        plt.savefig(os.path.join(output, "profile.png"))
        plt.close()

        # 2nd figure: objective vs iter
        def to_max(l):
            r = [l[0]]
            for e in l[1:]:
                r.append(max(r[-1], e))
            return r

        plt.figure()
        plt.plot(to_max(search.objective))
        plt.savefig(os.path.join(output, "search.png"))
        plt.close()


def _run_benchmark(script, output, verbose):

    script = os.path.abspath(script)
    l_script = script.split("/")
    i_src = l_script.index("src")
    output = os.path.join(output, *l_script[i_src+2:-1])

    logger.info(f"Loading benchmark from: {script}")

    sys.path.insert(0, os.path.dirname(script))
    module_name = os.path.basename(script)[:-3]
    mscript = importlib.import_module(module_name)

    def is_valid_benchmark(cls):
        try:
            b = cls is not Benchmark and issubclass(cls, Benchmark)
        except:
            b = False
        return b

    benchmark_classes = [
        getattr(mscript, attr)
        for attr in dir(mscript)
        if is_valid_benchmark(getattr(mscript, attr))
    ]

    assert len(benchmark_classes) == 1, f"Only 1 benchmark per script! Found {len(benchmark_classes)}..."

    output = os.path.join(output, mscript.__name__)
    logger.info(f"Saving report to: {output}")

    benchmark_class = benchmark_classes[0]

    benchmark = benchmark_class(verbose=verbose)
    benchmark.initialize()
    benchmark.execute()
    report = benchmark.report()

    _write_report(output, report)


def main():

    parser = _create_parser()
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())

    _run_benchmark(**vars(args))


if __name__ == "__main__":
    main()
