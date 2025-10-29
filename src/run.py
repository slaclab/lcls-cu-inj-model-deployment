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


def get_input_vars(model, interface_name):
    if interface_name == "test":
        # Use model input variable objects
        return model.input_variables
    elif interface_name == "epics" or interface_name == "k2eg":
        # Use PV names from the model's pv_map
        return model.input_names
    else:
        raise ValueError(f"Unknown interface: {interface_name}")


def run_iteration(model, interface, input_vars, interface_name, input_pv_transformer):
    """
    Run a single iteration of the SNDModel evaluation using the specified interface.

    Parameters
    ----------
    model : LUMEBaseModel
        The lume-model wrapped model instance to evaluate.
    interface : object
        The interface instance (TestInterface or EPICSInterface) for input retrieval.
    input_vars : list
        List of input variable names or PVs, depending on the interface.
    interface_name : str
        The name of the interface to use ('test' or 'epics').
    input_pv_transformer : InputPVTransformer
        The transformer to convert PV inputs to model inputs.

    Returns
    -------
    None
    """
    if interface_name == "test":
        # Get the input variable from the interface
        input_dict = interface.get_input_variables(input_vars)

    elif interface_name == "epics" or interface_name == "k2eg":
        # Get the input variables from the interface
        input_dict_raw = interface.get_input_variables(input_pv_transformer.input_list)
        logger.debug(f"Raw input values from EPICS: {MultiLineDict(input_dict_raw)}")
        # Get model inputs from PV inputs based on formulas defined in config.yaml
        input_dict = input_pv_transformer.transform(input_dict_raw)

        # TODO: adjust how this is done so it's standard for the models
        # TODO: validate that the model has PV names as the input names + any transforms
        # Map PVs back to model input names
        logger.debug(f"Transformed input values from EPICS: {MultiLineDict(input_dict)}")
        posixseconds = int(max(d["posixseconds"] for d in input_dict.values()))

        # Add constant Pulse_length, TODO: make this standard
        input_dict["Pulse_length"] = {"value": model.input_variables[1].default_value, "posixseconds": posixseconds}
        input_dict = {
            model.input_names[i]: input_dict[pv]["value"]
            for i, pv in enumerate(input_vars)
        }

    else:
        raise ValueError(f"Unknown interface: {interface_name}")

    logger.debug("Input values: %s", MultiLineDict(input_dict))

    # Evaluate the model with the input
    model.input_validation_config = {k: "warn" for k in model.input_names}
    if interface_name == "epics":
        # TODO: this was just for snd testing!!!
        # TODO: add transforms to base and remove this
        # Transform input from PV units to simulation units
        input_dict = model.input_transform(input_dict)
        logger.debug("Transformed input values: %s", MultiLineDict(input_dict))

    # Evaluate the model
    output = model.evaluate(input_dict)

    # Log input after transformation and output
    # one line to log at same timestamp
    # TODO: add epics timestamp to DB as well, and log all to wall clock time
    mlflow.log_metrics(
        input_dict | output,
        timestamp=(posixseconds * 1000 if interface_name == "epics" else None),
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
    input_vars = get_input_vars(model, args.interface)
    interface = get_interface(
        args.interface, input_vars if args.interface == "epics" else None
    )

    # Set up PV transformer
    with open("configs/pv_config.yaml", 'r') as f:
        config_yaml = yaml.safe_load(f)
    input_pv_transformer = InputPVTransformer(config_yaml)

    with MLflowRun() as run:
        # Log lockfile for complete reproducibility
        try:
            mlflow.log_artifact(PIXI_LOCKFILE_PATH, "pixi_lockfile")
        except FileNotFoundError:
            logger.error(f"Lockfile {PIXI_LOCKFILE_PATH} not found. Continuing without logging it.")
        # Run the evaluation loop
        while True:
            try:
                run_iteration(model, interface, input_vars, args.interface, input_pv_transformer)
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Exiting.")
                exit(0)
            except Exception as e:
                raise e


if __name__ == "__main__":
    main()
