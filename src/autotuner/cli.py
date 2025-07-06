import argparse
import os
import sys
from multiprocessing import cpu_count

import numpy as np

from autotuner.core.config import Settings, SweepConfig, TuneConfig, TuneEvalFunction
from autotuner.core.exceptions import ConfigurationError, FileOperationError, ValidationError
from autotuner.core.logging import get_logger
from autotuner.utils import prepare_ray_server, read_config

logger = get_logger(__name__)


def convert_to_settings(args_dict: dict) -> Settings:
    """Convert a dictionary of arguments to a Settings object."""
    logger.debug("Converting arguments to Settings object", extra={"args_keys": list(args_dict.keys())})

    try:
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
            logger.error("Unknown mode specified", extra={"mode": args_dict["mode"]})
            raise ConfigurationError(f"Unknown mode: {args_dict['mode']}")

        settings = Settings.model_validate(settings_dict)
        logger.info("Settings created successfully", extra={"mode": settings.mode, "design": settings.design})
        return settings
    except Exception as e:
        logger.error("Failed to convert arguments to Settings", exc_info=True)
        if isinstance(e, ConfigurationError | ValidationError):
            raise
        raise ConfigurationError(f"Failed to process configuration: {str(e)}") from e


def parse_arguments() -> Settings:
    """
    Parse arguments from command line.
    """
    logger.debug("Starting argument parsing")
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
        helper="Experiment name. This parameter is used to prefix the FLOW_VARIANT and to set the Ray log destination.",
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

    # Parse arguments
    args = parser.parse_args()
    args_dict = vars(args)
    logger.debug("Arguments parsed successfully", extra={"mode": args.mode, "design": args.design})

    # Process configuration file
    config_dict, SDC_ORIGINAL, FR_ORIGINAL = _process_config(args.config, args.mode, getattr(args, "algorithm", None))
    args_dict.update({"config_dict": config_dict, "sdc_original": SDC_ORIGINAL, "fr_original": FR_ORIGINAL})

    # Process paths
    LOCAL_DIR, ORFS_FLOW_DIR, INSTALL_PATH = _process_paths(args)
    args_dict.update({"local_dir": LOCAL_DIR, "orfs_flow_dir": ORFS_FLOW_DIR, "install_path": INSTALL_PATH})

    # Process reference file if needed
    if hasattr(args, "eval") and args.eval == TuneEvalFunction.PPA_IMPROV:
        reference_path = _process_reference(args.reference)
        args_dict["reference_dict"] = reference_path

    # Convert to settings and validate
    settings = convert_to_settings(args_dict)
    logger.info("Argument parsing completed successfully", extra={"experiment": settings.experiment})
    return settings


def _process_config(config_file: str, mode: str, algorithm: str | None = None) -> tuple:
    """Process configuration file and return config data."""
    try:
        config_dict, SDC_ORIGINAL, FR_ORIGINAL = read_config(
            file_name=os.path.abspath(config_file), mode=mode, algorithm=algorithm
        )
        logger.debug("Configuration processed successfully", extra={"config_file": config_file})
        return config_dict, SDC_ORIGINAL, FR_ORIGINAL
    except Exception as e:
        logger.error("Failed to process configuration file", extra={"config_file": config_file}, exc_info=True)
        raise ConfigurationError(f"Failed to process configuration file '{config_file}': {str(e)}") from e


def _process_paths(args) -> tuple:
    """Process Ray server paths."""
    try:
        LOCAL_DIR, ORFS_FLOW_DIR, INSTALL_PATH = prepare_ray_server(args)
        logger.debug("Paths processed successfully", extra={"local_dir": LOCAL_DIR, "orfs_flow_dir": ORFS_FLOW_DIR})
        return LOCAL_DIR, ORFS_FLOW_DIR, INSTALL_PATH
    except Exception as e:
        logger.error("Failed to prepare Ray server paths", exc_info=True)
        raise FileOperationError(f"Failed to prepare Ray server paths: {str(e)}") from e


def _process_reference(reference_file: str) -> str:
    """Process reference file for PPA improvement evaluation."""
    if not reference_file:
        logger.error("Reference file required for PPA improvement evaluation")
        raise ConfigurationError("Reference file is required when using PPA improvement evaluation")

    reference_path = os.path.abspath(reference_file)
    logger.debug("Reference file processed", extra={"reference_file": reference_file})
    return reference_path


def main():
    try:
        logger.info("Starting AutoTuner CLI")
        settings = parse_arguments()

        # Pretty print pydantic settings
        print(settings.model_dump_json(indent=2))
        logger.info("AutoTuner CLI completed successfully")
    except (ConfigurationError, ValidationError, FileOperationError) as e:
        logger.error(f"Configuration error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error in main", exc_info=True)
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
