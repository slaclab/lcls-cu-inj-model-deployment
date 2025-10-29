import argparse
import sys
import logging
import time
import collections
import yaml
import numpy as np
import mlflow
from mlflow_utils import MLflowRun, MLflowModelGetter
from configs.template_config import registered_model_name, model_version
from transformers.transformer import InputPVTransformer


logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

PIXI_LOCKFILE_PATH = "/app/pixi.lock"


class MultiLineDict(collections.UserDict):
    def __str__(self):
        return "\n" + "\n".join(f"{k} = {v}" for k, v in self.data.items())


def get_interface(interface_name, pvname_list=None):
    if interface_name == "test":
        from interface.test_interface import TestInterface

        return TestInterface()
    elif interface_name == "epics":
        from interface.epics_interface import EPICSInterface

        return EPICSInterface(pvname_list)
    elif interface_name == "k2eg":
        from interface.k2eg_interface import K2EGInterface

        return K2EGInterface()
    else:
        raise ValueError(f"Unknown interface: {interface_name}")


def run_iteration(model, interface, input_pv_transformer):
    """
    Run a single iteration of the SNDModel evaluation using the specified interface.

    Parameters
    ----------
    model : LUMEBaseModel
        The lume-model wrapped model instance to evaluate.
    interface : object
        The interface instance (TestInterface or EPICSInterface) for input retrieval.
    input_pv_transformer : InputPVTransformer
        The transformer to convert PV inputs to model inputs.

    Returns
    -------
    None
    """
    if interface.name == "test":
        # Get the input variable from the interface
        input_dict = interface.get_input_variables(model.input_variables)

    elif interface.name == "epics" or interface.name == "k2eg":
        # Get the values of input variables PVs from the interface
        input_dict_raw = interface.get_input_variables(input_pv_transformer.input_list)
        # Save the latest timestamp from EPICS PVs for logging
        max_posixseconds = int(max(d["posixseconds"] for d in input_dict_raw.values()))
        logger.debug(f"Raw input values from EPICS: {MultiLineDict(input_dict_raw)}")

        # Get model inputs from PV inputs based on formulas defined in config.yaml
        input_dict = input_pv_transformer.transform(input_dict_raw)
        logger.debug(f"Transformed input values from EPICS: {MultiLineDict(input_dict)}")

    else:
        raise ValueError(f"Unknown interface: {interface.name}")

    logger.debug("Input values: %s", MultiLineDict(input_dict))

    # Evaluate the model with the input
    # TODO: make this optional? or a config? for now, just warn on all inputs
    model.input_validation_config = {k: "warn" for k in model.input_names}
    # if interface.name == "epics":
    #     # TODO: this was just for snd testing!!!
    #     # TODO: add transforms to base and remove this
    #     # Transform input from PV units to simulation units
    #     input_dict = model.input_transform(input_dict)
    #     logger.debug("Transformed input values: %s", MultiLineDict(input_dict))

    # Evaluate the model
    output = model.evaluate(input_dict)

    # Log input after transformation and output
    # one line to log at same timestamp
    # TODO: add epics timestamp to DB as well, and log all to wall clock time
    mlflow.log_metrics(
        input_dict | output,
        timestamp=(max_posixseconds * 1000 if interface.name in ("epics", "k2eg") else None),
    )
    logger.debug("Output values: %s", MultiLineDict(output))


def main():
    """
    Main entry point for running the online model application with CLI interface selection.

    Parses command-line arguments to select the interface, initializes the model and interface,
    and runs the evaluation loop.

    You can run the script with:
        python run.py --interface test
        python run.py --interface epics

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Run the model with selected interface."
    )
    parser.add_argument(
        "--interface",
        "-i",
        choices=["test", "epics", "k2eg"],
        required=True,
        help="Interface to use",
    )
    args = parser.parse_args()
    logger.info("Running with interface: %s", args.interface)

    model = MLflowModelGetter(registered_model_name, model_version).get_model()

    # Set up PV transformer
    # This is required to map from EPICS PV names to model input names, and apply any formulas
    # defined in configs/pv_config.yaml. This is applicable only for EPICS/k2eg interfaces, and is in addition
    # to the lume-model's own internal input_transform method, if any are defined.
    with open("configs/pv_config.yaml", 'r') as f:
        config_yaml = yaml.safe_load(f)
    input_pv_transformer = InputPVTransformer(config_yaml)

    interface = get_interface(
        args.interface, input_pv_transformer.input_list if args.interface == "epics" else None
    )

    with MLflowRun() as run:
        # Log lockfile for complete reproducibility
        try:
            mlflow.log_artifact(PIXI_LOCKFILE_PATH, "pixi_lockfile")
        except FileNotFoundError:
            logger.error(f"Lockfile {PIXI_LOCKFILE_PATH} not found. Continuing without logging it.")
        # Run the evaluation loop
        while True:
            try:
                run_iteration(model, interface, input_pv_transformer)
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Exiting.")
                exit(0)
            except Exception as e:
                raise e


if __name__ == "__main__":
    main()
