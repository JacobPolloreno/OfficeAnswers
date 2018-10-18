import typing

from matchzoo import datapack
from matchzoo import engine


class DSSMUEPreprocessor(engine.BasePreprocessor):
    def __init__(self):
        self._datapack = None

    def fit(self,
            inputs: typing.List[tuple]):
        self._datapack = self.segmentation(inputs, stage='train')
        return self

    def transform(self,
                  inputs: typing.List[tuple],
                  stage: str,
                  cache: bool=True) -> datapack.DataPack:

        if cache:
            self._datapack = self.segmentation(inputs, stage='train')
            self._datapack.context['input_shapes'] = [(1,),
                                                      (1,)]
        else:
            datapack = self.segmentation(inputs, stage='train')
            datapack.context['input_shapes'] = [(1,),
                                                (1,)]

        return self._datapack if cache else datapack
