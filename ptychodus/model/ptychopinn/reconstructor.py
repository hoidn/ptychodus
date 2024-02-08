from __future__ import annotations
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping, TypeAlias
import logging

# TODO: Import ptychopinn specific classes once available
# from ptychopinn import SomeModelClass, SomeTesterClass, SomeTrainerClass
from scipy.ndimage import map_coordinates
import numpy
import numpy.typing

from ...api.image import ImageExtent
from ...api.object import ObjectArrayType, ObjectPatchAxis
from ...api.plot import Plot2D, PlotAxis, PlotSeries
from ...api.reconstructor import ReconstructInput, ReconstructOutput, TrainableReconstructor
from ..object import ObjectAPI
# TODO: Adjust settings import according to ptychopinn requirements
# from .settings import PtychoPinnModelSettings, PtychoPinnTrainingSettings

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
        # Assuming the first channel is phase and the second (if present) is amplitude
        self._buffer[self._pos, 0, :, :] = numpy.angle(array).astype(numpy.float32)

        if self._buffer.shape[1] > 1:
            self._buffer[self._pos, 1, :, :] = numpy.absolute(array).astype(numpy.float32)

        self._pos += 1

        if self._pos == self._buffer.shape[0]:
            self._pos = 0
            self._full = True

    def getBuffer(self) -> FloatArrayType:
        return self._buffer if self._full else self._buffer[:self._pos]

class PtychoPinnTrainableReconstructor(TrainableReconstructor):
    # Constructor and properties need to be adapted for ptychopinn specifics
    # TODO: Implement constructor with ptychopinn specific settings

    @property
    def name(self) -> str:
        # TODO: Return a meaningful name based on ptychopinn model specifics
        return 'PtychoPinnModelName'

    def _createModel(self) -> Any:
        # TODO: Implement model creation using ptychopinn specifics
        logger.debug('Building model...')
        # return SomeModelClass(...)

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        # Placeholder implementation. Specific ptychopinn reconstruction logic needed.
        logger.info("Reconstruction using ptychopinn model not yet implemented.")
        return ReconstructOutput.createNull()

    def ingestTrainingData(self, parameters: ReconstructInput) -> None:
        # Placeholder implementation. Specific ptychopinn data ingestion logic needed.
        logger.info("Ingesting training data for ptychopinn model not yet implemented.")

    def _plotMetrics(self, metrics: Mapping[str, Any]) -> Plot2D:
        # This method can likely remain unchanged, as it deals with plotting generic metrics
        # Implementation details here...

    def getSaveFileFilterList(self) -> Sequence[str]:
        # TODO: Adjust file filters according to ptychopinn's requirements
        return ['NumPy Zipped Archive (*.npz)']

    def getSaveFileFilter(self) -> str:
        # This method can likely remain unchanged, as it simply returns the first file filter
        return self._fileFilterList[0]

    def saveTrainingData(self, filePath: Path) -> None:
        # Placeholder implementation. Specific ptychopinn data saving logic needed.
        logger.info("Saving training data for ptychopinn model not yet implemented.")

    def train(self) -> Plot2D:
        # Placeholder implementation. Specific ptychopinn training logic needed.
        logger.info("Training using ptychopinn model not yet implemented.")
        return Plot2D.createNull()

    def clearTrainingData(self) -> None:
        # This method can likely remain unchanged, as it deals with resetting buffers
        # Implementation details here...
