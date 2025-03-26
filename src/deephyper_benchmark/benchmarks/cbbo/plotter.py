"""Module to run CBBO benchmarks."""

import argparse
import importlib
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tomllib
from deephyper.analysis import figure_size
from deephyper.analysis.hpo import (
    read_results_from_csv,
)


def get_attr_from_package(path):
    """Performs a relative import."""
    path = path.split(".")
    package = ".".join(path[:-1])
    name = path[-1]
    attr = importlib.import_module(name, package)
    return attr


def create_parser():
    """Creates the parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    return parser


class Plotter:
    def __init__(self, config_path):
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)
        self.root_path = os.getcwd()
        self.runs_path = os.path.join(self.root_path, "runs")
        self.plots_path = os.path.join(self.root_path, "plots")
        pathlib.Path(self.plots_path).mkdir(parents=False, exist_ok=True)

    def load_search_results(self, path):
        data = {}
        for replica_dir in os.scandir(path):
            results_path = os.path.join(replica_dir.path, "results.csv")
            if os.path.exists(results_path):
                data[replica_dir.name] = read_results_from_csv(results_path)
        return data

    def plot_average_simple_regret(self, bench, results, fig_path, y_scale_log=False):
        """Plot the average regret over replications."""
        scorer = bench.scorer

        fname = os.path.join(fig_path, "average_simple_regret")
        if y_scale_log:
            fname += "_log"
        fig, ax = plt.subplots(figsize=figure_size(width=600), tight_layout=True)

        x_min = 1
        x_max = self.config["search"]["max_evals"]
        kappa = 1.0
        legend = [[], []]
        y_min = None

        for i, search_label in enumerate(results):
            y = []
            y_item_len = None
            for replica_id in results[search_label]:
                obj = results[search_label][replica_id]["objective"].cummax().values
                regret = scorer.simple_regret(obj)
                if y_item_len is None:
                    y_item_len = len(regret)
                elif y_item_len != len(regret):
                    continue
                y.append(regret)
            y = np.asarray(y)
            x = np.arange(1, y.shape[1] + 1)

            color = f"C{i}"
            alpha = 0.2

            y_mean = np.mean(y, axis=0)
            y_stde = np.std(y, axis=0) / np.sqrt(len(y))
            y_low = y_mean - kappa * y_stde
            y_high = y_mean + kappa * y_stde

            # y_low = np.quantile(y, q=0.25, axis=0)
            # y_median = np.quantile(y, q=0.5, axis=0)
            # y_high = np.quantile(y, q=0.75, axis=0)

            # if y_min is None:
            #     y_min = np.min(y_low)
            # else:
            #     y_min = min(y_min, np.min(y_low))

            p1 = ax.plot(x, y_mean, color=color)
            ax.fill_between(
                x,
                y_low,
                y_high,
                alpha=alpha,
                color=color,
            )
            p2 = ax.fill(np.nan, np.nan, color=color, alpha=alpha)
            legend[0].append((p2[0], p1[0]))
            legend[1].append(search_label)

        ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min)

        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Regret")
        if y_scale_log:
            ax.set_yscale("log")
        ax.grid(visible=True, which="minor", linestyle=":")
        ax.grid(visible=True, which="major", linestyle="-")
        ax.legend(legend[0], legend[1])
        plt.savefig(fname, dpi=300)
        plt.close()

    def plot_average_cum_regret(self, bench, results, fig_path, y_scale_log=False):
        """Plot the average regret over replications."""
        scorer = bench.scorer

        fname = os.path.join(fig_path, "average_cumul_regret")
        if y_scale_log:
            fname += "_log"
        fig, ax = plt.subplots(figsize=figure_size(width=600), tight_layout=True)

        x_min = 1
        x_max = self.config["search"]["max_evals"]
        kappa = 1.0
        legend = [[], []]

        for i, search_label in enumerate(results):
            y = []
            y_item_len = None
            for replica_id in results[search_label]:
                obj = results[search_label][replica_id]["objective"].values
                regret = scorer.cumul_regret(obj)
                if y_item_len is None:
                    y_item_len = len(regret)
                elif y_item_len != len(regret):
                    continue
                y.append(regret)
            y = np.asarray(y)
            y_mean = np.mean(y, axis=0)
            y_stde = np.std(y, axis=0) / np.sqrt(len(y))
            x = np.arange(1, y.shape[1] + 1)

            color = f"C{i}"
            alpha = 0.2
            p1 = ax.plot(x, y_mean, color=color)
            ax.fill_between(
                x,
                y_mean - kappa * y_stde,
                y_mean + kappa * y_stde,
                alpha=alpha,
                color=color,
            )
            p2 = ax.fill(np.nan, np.nan, color=color, alpha=alpha)
            legend[0].append((p2[0], p1[0]))
            legend[1].append(search_label)

        ax.set_xlim(x_min, x_max)
        # ax.set_ylim(1e-6)

        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Cumulative Regret")
        if y_scale_log:
            ax.set_yscale("log")
        ax.grid(visible=True, which="minor", linestyle=":")
        ax.grid(visible=True, which="major", linestyle="-")
        ax.legend(legend[0], legend[1])
        plt.savefig(fname, dpi=300)
        plt.close()

    def plot_benchmark(self, label, config):
        """Plot a single benchmark."""
        self.runs_bench_path = os.path.join(self.runs_path, label)
        self.plots_bench_path = os.path.join(self.plots_path, label)
        pathlib.Path(self.plots_bench_path).mkdir(parents=False, exist_ok=True)

        bench = getattr(
            importlib.import_module(config["package"]),
            config["name"],
        )()

        data = {}
        for search_label, search_config in self.config["search"]["method"].items():
            data_path = os.path.join(self.runs_bench_path, search_label)
            if not os.path.exists(data_path):
                continue
            results = self.load_search_results(data_path)
            data[search_label] = results

        self.plot_average_simple_regret(bench, data, self.plots_bench_path)
        self.plot_average_simple_regret(bench, data, self.plots_bench_path, y_scale_log=True)
        self.plot_average_cum_regret(bench, data, self.plots_bench_path)

    def plot(self):
        """Plot everything."""
        for benchmark_label, benchmark_config in self.config["benchmark"].items():
            self.plot_benchmark(benchmark_label, benchmark_config)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    plotter = Plotter(args.config)
    plotter.plot()
