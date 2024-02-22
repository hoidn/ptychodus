from __future__ import annotations
from collections.abc import Sequence
from ..object import ObjectAPI
from pathlib import Path
from importlib.metadata import version
from .settings import PtychoPINNModelSettings, PtychoPINNTrainingSettings
from ...api.image import ImageExtent
from ...api.object import ObjectArrayType, ObjectPatchAxis
from ...api.plot import Plot2D, PlotAxis, PlotSeries
from ...api.reconstructor import ReconstructInput, ReconstructOutput, TrainableReconstructor
import numpy
import numpy.typing
import logging
from ptycho.loader import PtychoDataContainer

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

    def __init__(self, modelSettings: PtychoPINNModelSettings, trainingSettings: PtychoPINNTrainingSettings, objectAPI: ObjectAPI) -> None:
        self._modelSettings = modelSettings
        self._trainingSettings = trainingSettings
        self._objectAPI = objectAPI
        self._fileFilterList: list[str] = ['NumPy Zipped Archive (*.npz)']
        ptychopinnVersion = version('ptychopinn')
        logger.info(f'\tPtychoPINN {ptychopinnVersion}')
        self.modelSettings = modelSettings
        self.trainingSettings = trainingSettings
        ptychopinnVersion = version('ptychopinn')
        logger.info(f'\tPtychoPINN {ptychopinnVersion}')
        self._patternBuffer = PatternCircularBuffer.createZeroSized()
        self._objectPatchBuffer = ObjectPatchCircularBuffer.createZeroSized()
        self._fileFilterList = ['NumPy Zipped Archive (*.npz)']
        self.fileFilterList = ['NumPy Arrays (*.npy)', 'NumPy Zipped Archive (*.npz)']
        self._initialize_ptycho()
        self._ptychoDataContainer: PtychoDataContainer | None = None



    @property
    def name(self) -> str:
        return 'AmplitudePhase' 

    # Placeholder for the reconstruct method remains as implementing the actual logic requires details about the PtychoPINN model.

    def ingestTrainingData(self, parameters: ReconstructInput) -> None:
        # Extract diffraction patterns from the pattern buffer
        diffractionPatterns = self._patternBuffer.getBuffer()
        # Extract object patches (amplitude and phase) from the object patch buffer
        objectPatches = self._objectPatchBuffer.getBuffer()
        # Extract scan coordinates from the ReconstructInput parameter
        scanCoordinates = numpy.array(list(parameters.scan.values()))
        # This method will be updated in the next steps to use loader.RawData.from_coords_without_pc
        # to load a training dataset from the pattern buffer.
        from ptycho.loader import PtychoDataContainer
        diffractionPatterns = self._patternBuffer.getBuffer()
        scanCoordinates = numpy.array(list(parameters.scan.values()))
        xcoords, ycoords = scanCoordinates[:, 0], scanCoordinates[:, 1]
        probeGuess = parameters.probeArray
        objectGuess = parameters.objectInterpolator.getArray()
        self._ptychoDataContainer = PtychoDataContainer.from_raw_data_without_pc(
            xcoords=xcoords,
            ycoords=ycoords,
            diff3d=diffractionPatterns,
            probeGuess=probeGuess,
            scan_index=numpy.zeros(len(diffractionPatterns)),# TODO assuming all patches are from the same object
            objectGuess=objectGuess
        )

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self.fileFilterList

    def getSaveFileFilter(self) -> str:
        return self.fileFilterList[0]  # Default to the first option

    def saveTrainingData(self, filePath: Path) -> None:
        logger.debug(f'Writing \"{filePath}\" as \"NPZ\"')
        trainingData = {
            'diffractionPatterns': self._patternBuffer.getBuffer(),
            'objectPatches': self._objectPatchBuffer.getBuffer(),
        }
        numpy.savez(filePath, **trainingData)

    def _initialize_ptycho(self) -> None:
        from .params import update_cfg_from_settings, cfg
        from ptycho import params as ptycho_params
        # Update the configuration for ptycho based on the current settings in ptychodus
        update_cfg_from_settings(self.modelSettings)
        # Apply the updated configuration to ptycho's configuration
        ptycho_params.cfg.update(cfg)

    def train(self) -> Plot2D:
        if self._ptychoDataContainer is None:
            print("Training data has not been ingested. running _initialize_ptycho().")
            self._initialize_ptycho()
        else:
            print("Training data has already been ingested. Not running _initialize_ptycho().")
        from ptycho import train_pinn
        model_instance, history =  train_pinn.train(self._ptychoDataContainer)
        # TODO save the tensorflow model instance to file
        trainingLoss = history.history['loss']
        validationLoss = history.history['val_loss']  # Replace with actual validation loss values
        validationLossSeries = PlotSeries(label='Validation Loss', values=validationLoss)
        trainingLossSeries = PlotSeries(label='Training Loss', values=trainingLoss)
        seriesX = PlotSeries(label='Epoch', values=[*range(len(trainingLoss))])

        return Plot2D(
            axisX=PlotAxis(label='Epoch', series=[seriesX]),
            axisY=PlotAxis(label='Loss', series=[trainingLossSeries, validationLossSeries]),
        )

    def clearTrainingData(self) -> None:
        logger.debug('Clearing training data...')
        self._patternBuffer = PatternCircularBuffer.createZeroSized()
        self._objectPatchBuffer = ObjectPatchCircularBuffer.createZeroSized()
    def reconstruct(self, *args, **kwargs):
        raise NotImplementedError("Reconstruct method is not implemented yet.")
class PtychoPINNReconstructorLibrary:
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Load the model from the specified path
        self.model = self.load_model(model_path)

    def load_model(self, path: str):
        # Placeholder for model loading logic
        pass

    def reconstruct(self, input_data):
        # Placeholder for reconstruction logic using the loaded model
        pass

    def save_results(self, results, output_path: str):
        # Placeholder for saving the reconstruction results to a file
        pass
