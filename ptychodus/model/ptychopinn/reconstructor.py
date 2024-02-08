from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping, TypeAlias
import logging

import numpy
import numpy.typing

from ...api.image import ImageExtent
from ...api.object import ObjectArrayType, ObjectPatchAxis
from ...api.plot import Plot2D, PlotAxis, PlotSeries
from ...api.reconstructor import ReconstructInput, ReconstructOutput, TrainableReconstructor

FloatArrayType: TypeAlias = numpy.typing.NDArray[numpy.float32]

logger = logging.getLogger(__name__)

class PatternCircularBuffer:

    def __init__(self, extent: ImageExtent, maxSize: int) -> None:
        self._buffer: FloatArrayType = numpy.zeros(
            (maxSize, *extent.shape),
            dtype=numpy.float32,
        )
        self._pos = 0
        self._full = False

    @classmethod
    def createZeroSized(cls) -> PatternCircularBuffer:
        return cls(ImageExtent(0, 0), 0)

    @property
    def isZeroSized(self) -> bool:
        return (self._buffer.size == 0)

    def append(self, array: FloatArrayType) -> None:
        self._buffer[self._pos, :, :] = array
        self._pos += 1

        if self._pos == self._buffer.shape[0]:
            self._pos = 0
            self._full = True

    def getBuffer(self) -> FloatArrayType:
        return self._buffer if self._full else self._buffer[:self._pos]

class ObjectPatchCircularBuffer:

    def __init__(self, extent: ImageExtent, channels: int, maxSize: int) -> None:
        self._buffer: FloatArrayType = numpy.zeros(
            (maxSize, channels, *extent.shape),
            dtype=numpy.float32,
        )
        self._pos = 0
        self._full = False

    @classmethod
    def createZeroSized(cls) -> ObjectPatchCircularBuffer:
        return cls(ImageExtent(0, 0), 0, 0)

    @property
    def isZeroSized(self) -> bool:
        return (self._buffer.size == 0)

    def append(self, array: ObjectArrayType) -> None:
        self._buffer[self._pos, 0, :, :] = numpy.angle(array).astype(numpy.float32)

        if self._buffer.shape[1] > 1:
            self._buffer[self._pos, 1, :, :] = numpy.absolute(array).astype(numpy.float32)

        self._pos += 1

        if self._pos == self._buffer.shape[0]:
            self._pos = 0
            self._full = True

    def getBuffer(self) -> FloatArrayType:
        return self._buffer if self._full else self._buffer[:self._pos]
class PtychoPINNTrainableReconstructor(TrainableReconstructor):

    def __init__(self, modelSettings: Any, trainingSettings: Any) -> None:
        self.modelSettings = modelSettings
        self.trainingSettings = trainingSettings
        self.patternBuffer = PatternCircularBuffer.createZeroSized()
        self.objectPatchBuffer = ObjectPatchCircularBuffer.createZeroSized()
        self.patternBuffer = PatternCircularBuffer.createZeroSized()
        self.objectPatchBuffer = ObjectPatchCircularBuffer.createZeroSized()
        self.fileFilterList = ['NumPy Arrays (*.npy)', 'NumPy Zipped Archive (*.npz)']

    @property
    def name(self) -> str:
        return 'PtychoPINN'

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        # Placeholder for the reconstruction process
        return ReconstructOutput.createNull()

    def ingestTrainingData(self, parameters: ReconstructInput) -> None:
        if self.patternBuffer.isZeroSized:
            self.patternBuffer = PatternCircularBuffer(parameters.diffractionPatternExtent, self.trainingSettings.maximumTrainingDatasetSize)
        if self.objectPatchBuffer.isZeroSized:
            channels = 2  # Assuming amplitude and phase channels for PtychoPINN
            self.objectPatchBuffer = ObjectPatchCircularBuffer(parameters.objectExtent, channels, self.trainingSettings.maximumTrainingDatasetSize)
        for pattern in parameters.diffractionPatternArray:
            self.patternBuffer.append(pattern)
        for objectPatch in parameters.objectArray:  # Assuming objectArray contains patches
            self.objectPatchBuffer.append(objectPatch)

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self.fileFilterList

    def getSaveFileFilter(self) -> str:
        return self.fileFilterList[0]  # Default to the first option

    def saveTrainingData(self, filePath: Path) -> None:
        # TODO: Implement the logic to save the ingested training data to the specified file path
        # This may involve serialization and using the appropriate format based on the file extension
        raise NotImplementedError("Saving training data is not yet implemented.")

    def train(self) -> Plot2D:
        # TODO: Implement the training logic using the ingested training data
        # This should include model training, validation, and possibly early stopping
        # The method should return a Plot2D object representing the training progress, such as loss over epochs
        raise NotImplementedError("Training is not yet implemented.")
        # return Plot2D.createNull()  # Placeholder return statement

    def clearTrainingData(self) -> None:
        self.patternBuffer = PatternCircularBuffer.createZeroSized()
        self.objectPatchBuffer = ObjectPatchCircularBuffer.createZeroSized()
        self.patternBuffer = PatternCircularBuffer.createZeroSized()
        self.objectPatchBuffer = ObjectPatchCircularBuffer.createZeroSized()
