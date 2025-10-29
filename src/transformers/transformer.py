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
    """
    Transforms input PVs based on formulas defined in the configuration dictionary.

    The required configuration dictionary should have the following structure:
    {
        'input_variables': {
            'output_var1': {
                'formula': 'input_var1 + input_var2',
                'symbols': ['input_var1', 'input_var2']
                'proto': 'pva',
            },
            'output_var2': {
                'formula': 'input_var3 * 2',
                'symbols': ['input_var3']
                'proto': 'ca',
            },
            'output_var3': {
                'formula': '42',
            },
            ...
        }
    }

    For constant input PVs, the formula can simply be the constant value. Symbols should include the PV name,
    if any, used in the formula. The protocol ('proto') key can be used to specify the
    protocol for the output PV. This is optional and defaults to 'pva' if not provided.

    Methods
    -------
    transform(input_dict)
        Transforms the input PVs based on the defined formulas.
    """
    def __init__(self, config):
        """
        Initializes the InputPVTransformer with the given configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing input variable mappings and formulas.
        """
        pv_mapping = config['input_variables']
        # Get all symbols (PVs) used in the formulas
        self.input_list = []
        for c in pv_mapping:
            try:
                self.input_list.extend(pv_mapping[c]['symbols'])
            except KeyError:
                logger.debug(f"No symbols for {c}")

        logger.debug('Initializing Transformer')
        logger.debug(f'PV Mapping: {pv_mapping}')
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
        """
        Transforms the input PVs based on the defined formulas in the config.

        Parameters
        ----------
        input_dict: dict
            Dictionary mapping PV names to their values and timestamps.

        Returns
        -------
        dict
            Dictionary mapping transformed variable names to their computed values and timestamps.
        """
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
        transformed = {}
        pvs_renamed = {
            key.replace(':', '_'): value["value"] for key, value in input_dict.items()
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