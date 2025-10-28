import logging
import sys
import numpy as np
import sympy as sp
import yaml

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


class InputPVTransformer():
    def __init__(self, config):
        pv_mapping = config['input_variables']
        # Get all symbols
        self.input_list = []
        for c in pv_mapping:
            try:
                self.input_list.extend(pv_mapping[c]['symbols'])
            except KeyError:
                print(f"no symbols for {c}")

        logger.debug('Initializing Transformer')
        logger.debug(f'PV Mapping: {pv_mapping}') # pv mapping would be the input variable names of the model (main keys)
        logger.debug(f'Symbol List: {self.input_list}')
        self.pv_mapping = pv_mapping

        for key, value in self.pv_mapping.items():
            self._validate_formulas(str(value['formula']))
        self.formulas = {}
        self.lambdified_formulas = {}
        for key, value in self.pv_mapping.items():
            self.formulas[key] = sp.sympify(str(value['formula']).replace(':', '_'))
            input_list_renamed = [
                symbol.replace(':', '_') for symbol in self.input_list
            ]
            self.lambdified_formulas[key] = sp.lambdify(
                input_list_renamed, self.formulas[key], modules='numpy'
            )

    def _validate_formulas(self, formula: str):
        try:
            sp.sympify(formula.replace(':', '_'))
        except Exception as e:
            raise Exception(f'Invalid formula: {formula}: {e}')

    def transform(self, input_dict):
        for pv, value in input_dict.items():
            # assert value is float
            try:
                if isinstance(value["value"], (float, int, np.float32)):
                    value["value"] = float(value["value"])
                elif isinstance(value["value"], (np.ndarray, list)):
                    value["value"] = np.array(value["value"]).astype(float)
                else:
                    raise Exception(
                        f'Invalid type for value: {value["value"]}, type: {type(value["value"])}'
                    )
            except Exception as e:
                logger.error(f'Error converting value to float: {e}')
                raise e

        try:
            return self._transform(input_dict)
        except Exception as e:
            logger.error(f'Error transforming: {e}')
            raise e

    def _transform(self, input_dict):
        transformed = {key: {"value": None, "posixseconds": None} for key in self.pv_mapping.keys()}
        timestamps = [input_dict[key]["posixseconds"] for key in input_dict.keys()]
        max_timestamp = max(timestamps)
        pvs_renamed = {
            key.replace(':', '_'): value["value"] for key, value in input_dict.items()
        }

        for key in self.pv_mapping.keys():
            try:
                lambdified_formula = self.lambdified_formulas[key]
                transformed[key]["value"] = lambdified_formula(*[
                    pvs_renamed[symbol.replace(':', '_')] for symbol in self.input_list
                ])

                if isinstance(transformed[key]["value"], np.ndarray):
                    if transformed[key]["value"].shape[-1] == 1:
                        transformed[key]["value"] = transformed[key]["value"].squeeze()
                else:
                    transformed[key]["value"] = float(transformed[key]["value"])

                # Preserve timestamp from input
                try:
                    transformed[key]["posixseconds"] = input_dict[key]["posixseconds"]
                except KeyError:
                    # TODO: actually set it to the max of the input PVs used in the formula
                    transformed[key]["posixseconds"] = max_timestamp

            except Exception as e:
                logger.error(f'Error transforming: {e}')
                raise e

        return transformed