import argparse
from multiprocessing import cpu_count

import numpy as np

from autotuner.core.config import Settings, SweepConfig, TuneConfig


def convert_to_settings(args_dict: dict):
    tune_keys = TuneConfig.model_fields.keys()
    sweep_keys = SweepConfig.model_fields.keys()

    settings_dict = {
        "design": args_dict["design"],
        "platform": args_dict["platform"],
        "experiment": args_dict["experiment"],
        "timeout": args_dict["timeout"],
        "verbose": args_dict["verbose"],
        "jobs": args_dict["jobs"],
        "openroad_threads": args_dict["openroad_threads"],
        "server": args_dict["server"],
        "port": args_dict["port"],
        "mode": args_dict["mode"],
    }
    if args_dict["mode"] == "tune":
        settings_dict["tune"] = {key: args_dict[key] for key in tune_keys if key in args_dict}
    elif args_dict["mode"] == "sweep":
        settings_dict["sweep"] = {key: args_dict[key] for key in sweep_keys if key in args_dict}
    else:
        raise ValueError(f"Unknown mode: {args_dict['mode']}")

    settings = Settings.model_validate(settings_dict)
    return settings


def parse_arguments():
    """
    Parse arguments from command line.
    """
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help="mode of execution", dest="mode", required=True)
    tune_parser = subparsers.add_parser("tune")
    _ = subparsers.add_parser("sweep")

    # DUT
    parser.add_argument(
        "--design",
        type=str,
        metavar="<gcd,jpeg,ibex,aes,...>",
        required=True,
        help="Name of the design for Autotuning.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        metavar="<sky130hd,sky130hs,asap7,...>",
        required=True,
        help="Name of the platform for Autotuning.",
    )

    # Experiment Setup
    parser.add_argument(
        "--config",
        type=str,
        metavar="<path>",
        required=True,
        help="Configuration file that sets which knobs to use for Autotuning.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        metavar="<str>",
        default="test",
        help="Experiment name. This parameter is used to prefix the FLOW_VARIANT and to set the Ray log destination.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        metavar="<float>",
        default=None,
        help="Time limit (in hours) for each trial run. Default is no limit.",
    )
    tune_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous run. Note that you must also set a unique experiment\
                name identifier via `--experiment NAME` to be able to resume.",
    )

    # ML
    tune_parser.add_argument(
        "--algorithm",
        type=str,
        choices=["hyperopt", "ax", "optuna", "pbt", "random"],
        default="hyperopt",
        help="Search algorithm to use for Autotuning.",
    )
    tune_parser.add_argument(
        "--eval",
        type=str,
        choices=["default", "ppa-improv"],
        default="default",
        help="Evaluate function to use with search algorithm.",
    )
    tune_parser.add_argument(
        "--samples",
        type=int,
        metavar="<int>",
        default=10,
        help="Number of samples for tuning.",
    )
    tune_parser.add_argument(
        "--iterations",
        type=int,
        metavar="<int>",
        default=1,
        help="Number of iterations for tuning.",
    )
    tune_parser.add_argument(
        "--resources_per_trial",
        type=float,
        metavar="<float>",
        default=1,
        help="Number of CPUs to request for each tuning job.",
    )
    tune_parser.add_argument(
        "--reference",
        type=str,
        metavar="<path>",
        default=None,
        help="Reference file for use with PPAImprov.",
    )
    tune_parser.add_argument(
        "--perturbation",
        type=int,
        metavar="<int>",
        default=25,
        help="Perturbation interval for PopulationBasedTraining.",
    )
    tune_parser.add_argument(
        "--seed",
        type=int,
        metavar="<int>",
        default=42,
        help="Random seed. (0 means no seed.)",
    )

    # Workload
    parser.add_argument(
        "--jobs",
        type=int,
        metavar="<int>",
        default=int(np.floor(cpu_count() / 2)),
        help="Max number of concurrent jobs.",
    )
    parser.add_argument(
        "--openroad_threads",
        type=int,
        metavar="<int>",
        default=16,
        help="Max number of threads openroad can use.",
    )
    parser.add_argument(
        "--server",
        type=str,
        metavar="<ip|servername>",
        default=None,
        help="The address of Ray server to connect.",
    )
    parser.add_argument(
        "--port",
        type=int,
        metavar="<int>",
        default=10001,
        help="The port of Ray server to connect.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity level.\n\t0: only print Ray status\n\t1: also print"
        " training stderr\n\t2: also print training stdout.",
    )

    args = parser.parse_args()

    # Validate using Pydantic!
    args_dict = vars(args)
    settings = convert_to_settings(args_dict)

    return settings


def main():
    settings = parse_arguments()

    # Pretty print pydantic settings
    print(settings.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
