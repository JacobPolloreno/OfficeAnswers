import numpy as np

from matchzoo import engine
from matchzoo import preprocessor


class Hasher(object):
    def __init__(self,
                 preprocessor_dir: str,
                 preprocessor_name: str) -> None:
        self._pre = engine.load_preprocessor(preprocessor_dir,
                                             preprocessor_name)
        self._hasher = preprocessor.WordHashingUnit(
            self._pre._context['term_index'])

    def get_sparse_input(self, text: str) -> np.ndarray:
        return self._hasher.transform(text)
