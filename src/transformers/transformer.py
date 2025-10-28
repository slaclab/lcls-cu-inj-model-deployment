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
                if isinstance(value, (float, int, np.float32)):
                    value = float(value)
                elif isinstance(value, (np.ndarray, list)):
                    value = np.array(value).astype(float)
                else:
                    raise Exception(
                        f'Invalid type for value: {value}, type: {type(value)}'
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
        transformed = {}
        pvs_renamed = {
            key.replace(':', '_'): value for key, value in input_dict.items()
        }

        for key in self.pv_mapping.keys():
            try:
                lambdified_formula = self.lambdified_formulas[key]
                transformed[key] = lambdified_formula(*[
                    pvs_renamed[symbol.replace(':', '_')] for symbol in self.input_list
                ])

                if isinstance(transformed[key], np.ndarray):
                    if transformed[key].shape[-1] == 1:
                        transformed[key] = transformed[key].squeeze()
                else:
                    transformed[key] = float(transformed[key])

            except Exception as e:
                logger.error(f'Error transforming: {e}')
                raise e

        return transformed