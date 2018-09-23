"""DRMM Preprocessor"""

import typing
import logging

from matchzoo import datapack
from matchzoo import engine
from matchzoo import preprocessor
from tqdm import tqdm


logger = logging.getLogger(__name__)


class DRMMPreprocessor(engine.BasePreprocessor):
    """
    DRMM Preprocessor

    Example:

    """

    def __init__(self):
        """Initilization."""
        pass

    def _prepare_stateless_units(self) -> list:
        """Prepare needed process units."""
        pass
