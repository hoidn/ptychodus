from __future__ import annotations
from collections.abc import Sequence
from importlib.metadata import version
from typing import Any, TypeAlias
import numpy.typing

from ...api.image import ImageExtent
from ...api.object import ObjectArrayType, ObjectPatchAxis
from ...api.plot import Plot2D, PlotAxis, PlotSeries
from ...api.reconstructor import ReconstructInput, ReconstructOutput, TrainableReconstructor
from ..object import ObjectAPI
from .settings import PtychoPINNModelSettings, PtychoPINNTrainingSettings

FloatArrayType: TypeAlias = numpy.typing.NDArray[numpy.float32]
from pathlib import Path
from typing import Any, Mapping
import logging

import numpy
import numpy.typing

from ...api.image import ImageExtent
from ...api.object import ObjectArrayType, ObjectPatchAxis
from ...api.plot import Plot2D, PlotAxis, PlotSeries
from ...api.reconstructor import ReconstructInput, ReconstructOutput, TrainableReconstructor

FloatArrayType = numpy.typing.NDArray[numpy.float32]

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

    def __init__(self, modelSettings: PtychoPINNModelSettings, trainingSettings: PtychoPINNTrainingSettings, objectAPI: ObjectAPI, *, enableAmplitude: bool) -> None:
        self._modelSettings = modelSettings
        self._trainingSettings = trainingSettings
        self._objectAPI = objectAPI
        self._enableAmplitude = enableAmplitude
        self._fileFilterList: list[str] = ['NumPy Zipped Archive (*.npz)']
        ptychopinnVersion = version('ptychopinn')
        logger.info(f'\tPtychoPINN {ptychopinnVersion}')
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

    # Placeholder for the reconstruct method remains as implementing the actual logic requires details about the PtychoPINN model.

    def ingestTrainingData(self, parameters: ReconstructInput) -> None:
        # Adjusted to match the API specification and example implementation. Actual logic depends on the model details.
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
        logger.debug(f'Writing \"{filePath}\" as \"NPZ\"')
        trainingData = {
            'diffractionPatterns': self.patternBuffer.getBuffer(),
            'objectPatches': self.objectPatchBuffer.getBuffer(),
        }
        numpy.savez(filePath, **trainingData)

    def train(self) -> Plot2D:
        # Detailed TODO: Implement the model training logic specific to PtychoPINN
        # This should include initializing the model, preparing the data, running the training loop,
        # and validating the model. The specifics of these steps depend on the PtychoPINN architecture
        # and training procedure, which are not detailed here.
        #
        # After training, generate a Plot2D object to visualize the training progress, such as loss over epochs.
        # This visualization is crucial for understanding the training dynamics and evaluating the model's performance.
        #
        # Placeholder for training logic:
        # Initialize model, prepare data, run training loop, validate model
        #
        # Placeholder for generating Plot2D object:
        trainingLoss = [0]  # Replace with actual training loss values
        validationLoss = [0]  # Replace with actual validation loss values
        validationLossSeries = PlotSeries(label='Validation Loss', values=validationLoss)
        trainingLossSeries = PlotSeries(label='Training Loss', values=trainingLoss)
        seriesX = PlotSeries(label='Epoch', values=[*range(len(trainingLoss))])

        return Plot2D(
            axisX=PlotAxis(label='Epoch', series=[seriesX]),
            axisY=PlotAxis(label='Loss', series=[trainingLossSeries, validationLossSeries]),
        )

    def clearTrainingData(self) -> None:
        self.patternBuffer = PatternCircularBuffer.createZeroSized()
        self.objectPatchBuffer = ObjectPatchCircularBuffer.createZeroSized()
        self.patternBuffer = PatternCircularBuffer.createZeroSized()
        self.objectPatchBuffer = ObjectPatchCircularBuffer.createZeroSized()
        self.patternBuffer = PatternCircularBuffer.createZeroSized()
        self.objectPatchBuffer = ObjectPatchCircularBuffer.createZeroSized()
