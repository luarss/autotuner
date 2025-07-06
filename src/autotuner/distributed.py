#############################################################################
##
## BSD 3-Clause License
##
## Copyright (c) 2019, The Regents of the University of California
## All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
## * Redistributions of source code must retain the above copyright notice, this
##   list of conditions and the following disclaimer.
##
## * Redistributions in binary form must reproduce the above copyright notice,
##   this list of conditions and the following disclaimer in the documentation
##   and/or other materials provided with the distribution.
##
## * Neither the name of the copyright holder nor the names of its
##   contributors may be used to endorse or promote products derived from
##   this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
## ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
## LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
## CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
## SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
## INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
## CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
## ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
## POSSIBILITY OF SUCH DAMAGE.
##
###############################################################################

"""
This scripts handles sweeping and tuning of OpenROAD-flow-scripts parameters.
Dependencies are documented in pip format at distributed-requirements.txt

For both sweep and tune modes:
    openroad_autotuner -h

Note: the order of the parameters matter.
Arguments --design, --platform and --config are always required and should
precede the <mode>.

AutoTuner:
    openroad_autotuner tune -h
    openroad_autotuner --design gcd --platform sky130hd \
                           --config ../designs/sky130hd/gcd/autotuner.json \
                           tune
    Example:

Parameter sweeping:
    openroad_autotuner sweep -h
    Example:
    openroad_autotuner --design gcd --platform sky130hd \
                       --config distributed-sweep-example.json \
                       sweep
"""

import json
import os
import random
import sys
from collections import namedtuple
from itertools import product
from multiprocessing import cpu_count

import numpy as np
import ray
import torch
from ax.service.ax_client import AxClient
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, PopulationBasedTraining
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.ax import AxSearch
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.util.queue import Queue

from autotuner.cli import parse_arguments
from autotuner.core.config import Settings
from autotuner.core.exceptions import (
    DistributedExecutionError,
)
from autotuner.core.logging import get_logger
from autotuner.utils import (
    consumer,
    openroad,
    parse_config,
    read_metrics,
)

logger = get_logger(__name__)


class AutoTunerBase(tune.Trainable):
    """
    AutoTuner base class for experiments.
    """

    def setup(self, config: dict, settings: Settings | None = None):
        """
        Setup current experiment step.
        """
        logger.debug("Setting up AutoTuner experiment", extra={"trial_id": getattr(self, "trial_id", "unknown")})

        try:
            # We create the following directory structure:
            #      1/     2/         3/       4/           5/
            # <repo>/<logs>/<platform>/<design>/<experiment/<cwd>
            if settings is None:
                logger.error("Settings not provided to AutoTuner setup")
                raise DistributedExecutionError("[ERROR TUN-0004] Settings must be provided.")

            self.settings = settings
            local_dir_str = str(self.settings.local_dir) if self.settings.local_dir is not None else ""
            self.repo_dir = os.path.abspath(os.path.join(local_dir_str, *[".."] * 4))

            self.parameters = parse_config(
                config=config,
                platform=self.settings.platform,
                sdc_original=self.settings.sdc_original,
                constraints_sdc=self.settings.constraints_sdc,
                fr_original=self.settings.fr_original,
                fastroute_tcl=self.settings.fastroute_tcl,
                path=os.getcwd(),
            )

            self.step_ = 0
            self.variant = f"variant-{self.__class__.__name__}-{self.trial_id}-or"

            # Do a valid config check here, since we still have the config in a
            # dict vs. having to scan through the parameter string later
            self.is_valid_config = self._is_valid_config(config)

            logger.info(
                "AutoTuner experiment setup completed",
                extra={
                    "trial_id": self.trial_id,
                    "variant": self.variant,
                    "valid_config": self.is_valid_config,
                    "platform": self.settings.platform,
                    "design": self.settings.design,
                },
            )

        except Exception as e:
            logger.error("Failed to setup AutoTuner experiment", exc_info=True)
            if isinstance(e, DistributedExecutionError):
                raise
            raise DistributedExecutionError(f"Failed to setup experiment: {str(e)}") from e

    def step(self):
        """
        Run step experiment and compute its score.
        """

        # if not a valid config, then don't run and pass back an error
        if not self.is_valid_config:
            return {self.settings.metric: self.settings.error_metric, "effective_clk_period": "-", "num_drc": "-"}
        self._variant = f"{self.variant}-{self.step_}"
        metrics_file = openroad(
            args=self.settings,
            base_dir=self.repo_dir,
            parameters=self.parameters,
            flow_variant=self._variant,
            install_path=self.settings.install_path,
        )
        self.step_ += 1
        (score, effective_clk_period, num_drc) = self.evaluate(read_metrics(metrics_file))
        # Feed the score back to Tune.
        # return must match 'metric' used in tune.run()
        return {
            self.settings.metric: score,
            "effective_clk_period": effective_clk_period,
            "num_drc": num_drc,
        }

    def evaluate(self, metrics):
        """
        User-defined evaluation function.
        It can change in any form to minimize the score (return value).
        Default evaluation function optimizes effective clock period.
        """
        error = "ERR" in metrics.values()
        not_found = "N/A" in metrics.values()
        if error or not_found:
            return (self.settings.error_metric, "-", "-")
        effective_clk_period = metrics["clk_period"] - metrics["worst_slack"]
        num_drc = metrics["num_drc"]
        gamma = effective_clk_period / 10
        score = effective_clk_period
        score = score * (100 / self.step_) + gamma * num_drc
        return (score, effective_clk_period, num_drc)

    def _is_valid_config(self, config):
        """
        Checks dependent parameters and returns False if we violate
        a dependency. That way, we don't end up running an incompatible run
        """

        ret_val = True
        ret_val &= self._is_valid_padding(config)
        return ret_val

    def _is_valid_padding(self, config):
        """Returns True if global padding >= detail padding"""

        if "CELL_PAD_IN_SITES_GLOBAL_PLACEMENT" in config and "CELL_PAD_IN_SITES_DETAIL_PLACEMENT" in config:
            global_padding = config["CELL_PAD_IN_SITES_GLOBAL_PLACEMENT"]
            detail_padding = config["CELL_PAD_IN_SITES_DETAIL_PLACEMENT"]
            if global_padding < detail_padding:
                print(
                    f"[WARN TUN-0032] CELL_PAD_IN_SITES_DETAIL_PLACEMENT ({detail_padding}) cannot be greater than CELL_PAD_IN_SITES_GLOBAL_PLACEMENT ({global_padding})"
                )
                return False
        return True


class PPAImprov(AutoTunerBase):
    """
    PPAImprov
    """

    def setup(self, config: dict, settings: Settings | None = None):
        """Load reference metrics during setup."""
        super().setup(config)
        if settings is None:
            raise ValueError("[ERROR TUN-0005] Settings must be provided.")
        self.settings = settings

    def get_ppa(self, metrics: dict):
        """
        Compute PPA term for evaluate.
        """
        coeff_perform, coeff_power, coeff_area = 10000, 100, 100

        eff_clk_period = metrics["clk_period"]
        if metrics["worst_slack"] < 0:
            eff_clk_period -= metrics["worst_slack"]

        eff_clk_period_ref = self.settings.tune.reference_dict["clk_period"]
        if self.settings.tune.reference_dict["worst_slack"] < 0:
            eff_clk_period_ref -= self.settings.tune.reference_dict["worst_slack"]

        def percent(x_1, x_2):
            return (x_1 - x_2) / x_1 * 100

        performance = percent(eff_clk_period_ref, eff_clk_period)
        power = percent(self.settings.tune.reference_dict["total_power"], metrics["total_power"])
        area = percent(100 - self.settings.tune.reference_dict["final_util"], 100 - metrics["final_util"])

        # Lower values of PPA are better.
        ppa_upper_bound = (coeff_perform + coeff_power + coeff_area) * 100
        ppa = performance * coeff_perform
        ppa += power * coeff_power
        ppa += area * coeff_area
        return ppa_upper_bound - ppa

    def evaluate(self, metrics):
        error = "ERR" in metrics.values() or "ERR" in self.settings.tune.reference_dict.values()
        not_found = "N/A" in metrics.values() or "N/A" in self.settings.tune.reference_dict.values()
        if error or not_found:
            return (self.settings.error_metric, "-", "-")
        ppa = self.get_ppa(metrics)
        gamma = ppa / 10
        score = ppa * (self.step_ / 100) ** (-1) + (gamma * metrics["num_drc"])
        effective_clk_period = metrics["clk_period"] - metrics["worst_slack"]
        num_drc = metrics["num_drc"]
        return (score, effective_clk_period, num_drc)


def set_algorithm(args, best_params):
    """
    Configure search algorithm.
    """
    # Pre-set seed if user sets seed to 0
    if args.seed == 0:
        print("Warning: you have chosen not to set a seed. Do you wish to continue? (y/n)")
        if input().lower() != "y":
            sys.exit(0)
        args.seed = None
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.tune.algorithm == "hyperopt":
        algorithm = HyperOptSearch(
            points_to_evaluate=best_params,
            random_state_seed=args.seed,
        )
    elif args.tune.algorithm == "ax":
        ax_client = AxClient(
            enforce_sequential_optimization=False,
            random_seed=args.seed,
        )
        AxClientMetric = namedtuple("AxClientMetric", "minimize")
        ax_client.create_experiment(
            name=args.experiment,
            parameters=args.tune.config_dict,
            objectives={args.metric: AxClientMetric(minimize=True)},
        )
        algorithm = AxSearch(ax_client=ax_client, points_to_evaluate=best_params)
    elif args.tune.algorithm == "optuna":
        algorithm = OptunaSearch(points_to_evaluate=best_params, seed=args.seed)
    elif args.tune.algorithm == "pbt":
        print("Warning: PBT does not support seed values. seed will be ignored.")
        algorithm = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=args.tune.perturbation,
            hyperparam_mutations=args.tune.config_dict,
            synch=True,
        )
    elif args.tune.algorithm == "random":
        algorithm = BasicVariantGenerator(
            max_concurrent=args.jobs,
            random_state=args.seed,
        )
    else:
        raise ValueError(
            f"[ERROR TUN-0001] Unknown algorithm {args.tune.algorithm}. "
            "Supported algorithms: hyperopt, ax, optuna, pbt, random."
        )

    # A wrapper algorithm for limiting the number of concurrent trials.
    if args.tune.algorithm not in ["random", "pbt"]:
        algorithm = ConcurrencyLimiter(algorithm, max_concurrent=args.jobs)

    return algorithm


def set_best_params(platform, design):
    """
    Get current known best parameters if it exists.
    """
    params = []
    best_param_file = f"designs/{platform}/{design}/autotuner-best.json"
    if os.path.isfile(best_param_file):
        with open(best_param_file) as file:
            params = json.load(file)
    return params


def set_training_class(settings: Settings):
    """
    Set training class.
    """
    if settings.tune.eval == "default":
        return tune.with_parameters(AutoTunerBase, settings=settings)
    elif settings.tune.eval == "ppa-improv":
        return tune.with_parameters(PPAImprov, settings=settings)
    return None


# TODO: Make this remote function take in arguments.
@ray.remote
def save_best(results):
    """
    Save best configuration of parameters found.
    """
    best_config = results.best_config
    best_config["best_result"] = results.best_result[METRIC]
    trial_id = results.best_trial.trial_id
    new_best_path = f"{LOCAL_DIR}/{args.experiment}/"
    new_best_path += f"autotuner-best-{trial_id}.json"
    with open(new_best_path, "w") as new_best_file:
        json.dump(best_config, new_best_file, indent=4)
    print(f"[INFO TUN-0003] Best parameters written to {new_best_path}")


def sweep(settings: Settings):
    """Run sweep of parameters"""
    if settings.server is not None:
        # For remote sweep we create the following directory structure:
        #      1/     2/         3/       4/
        # <repo>/<logs>/<platform>/<design>/
        repo_dir = os.path.abspath(str(settings.local_dir) + "/../" * 4)
    else:
        repo_dir = os.path.abspath(os.path.join(str(settings.orfs_flow_dir), ".."))
    print(f"[INFO TUN-0012] Log folder {str(settings.local_dir)}.")
    queue = Queue()
    parameter_list = list()
    for name, content in settings.tune.config_dict.items():
        if not isinstance(content, list):
            print(f"[ERROR TUN-0015] {name} sweep is not supported.")
            sys.exit(1)
        if content[-1] == 0:
            print("[ERROR TUN-0014] Sweep does not support step value zero.")
            sys.exit(1)
        parameter_list.append([{name: i} for i in np.arange(*content)])
    parameter_list = list(product(*parameter_list))
    for parameter in parameter_list:
        temp = dict()
        for value in parameter:
            temp.update(value)
        queue.put([settings, repo_dir, temp, settings.sdc_original, settings.fr_original, settings.install_path])
    workers = [consumer.remote(queue) for _ in range(settings.jobs)]
    print("[INFO TUN-0009] Waiting for results.")
    ray.get(workers)
    print("[INFO TUN-0010] Sweep complete.")


def main():
    # TODO -rename args to settings everywhere.
    args = parse_arguments()

    if args.mode == "tune":
        best_params = set_best_params(args.platform, args.design)
        search_algo = set_algorithm(args, best_params)
        TrainClass = set_training_class(args)
        tune_args = dict(
            name=args.experiment,
            metric=args.metric,
            mode="min",
            num_samples=args.tune.samples,
            fail_fast=False,
            storage_path=args.local_dir,
            resume=args.tune.resume,
            stop={"training_iteration": args.tune.iterations},
            resources_per_trial={"cpu": cpu_count() / args.jobs},
            log_to_file=["trail-out.log", "trail-err.log"],
            trial_name_creator=lambda x: f"variant-{x.trainable_name}-{x.trial_id}-ray",
            trial_dirname_creator=lambda x: f"variant-{x.trainable_name}-{x.trial_id}-ray",
            settings=args,
        )
        if args.tune.algorithm == "pbt":
            os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(args.jobs)
            tune_args["scheduler"] = search_algo
        else:
            tune_args["search_alg"] = search_algo
            tune_args["scheduler"] = AsyncHyperBandScheduler()
        if args.tune.algorithm != "ax":
            tune_args["config"] = args.tune.config_dict
        analysis = tune.run(TrainClass, **tune_args)

        task_id = save_best.remote(analysis)
        _ = ray.get(task_id)
        print(f"[INFO TUN-0002] Best parameters found: {analysis.best_config}")

        # if all runs have failed
        if analysis.best_result[args.metric] == args.error_metric:
            print("[ERROR TUN-0016] No successful runs found.")
            sys.exit(16)
    elif args.mode == "sweep":
        sweep(args)


if __name__ == "__main__":
    main()
