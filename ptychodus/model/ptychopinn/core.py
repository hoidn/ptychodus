from collections.abc import Iterator, Sequence
from ...api.reconstructor import Reconstructor, ReconstructorLibrary
from ...api.settings import SettingsRegistry
from .settings import PtychoPINNModelSettings, PtychoPINNTrainingSettings
from ...api.geometry import Interval
from ...api.observer import Observer
from .reconstructor import PtychoPINNTrainableReconstructor

class PtychoPINNModelPresenter(Observer):
    def __init__(self, settings: PtychoPINNModelSettings) -> None:
        self._settings = settings

    # Add methods to interact with individual settings here
    # Similar to those in PtychoNNModelPresenter
    # Example:
    def getLearningRate(self) -> float:
        return float(self._settings.learningRate.value)

    def setLearningRate(self, value: float) -> None:
        self._settings.learningRate.value = value

    def getN(self) -> int:
        return self._settings.N.value

    def setN(self, value: int) -> None:
        self._settings.N.value = value

    def getGridsize(self) -> int:
        return self._settings.gridsize.value

    def setGridsize(self, value: int) -> None:
        self._settings.gridsize.value = value

    def getBatchSize(self) -> int:
        return self._settings.batch_size.value

    def setBatchSize(self, value: int) -> None:
        self._settings.batch_size.value = value

    # ... methods for other settings continue ...

class PtychoPINNTrainingPresenter(Observer):
    def __init__(self, settings: PtychoPINNTrainingSettings) -> None:
        self._settings = settings

    # Add methods to interact with individual settings here
    # Similar to those in PtychoNNTrainingPresenter
    # Example:
    def getMaximumTrainingDatasetSize(self) -> int:
        return self._settings.maximumTrainingDatasetSize.value

    def setMaximumTrainingDatasetSize(self, value: int) -> None:
        self._settings.maximumTrainingDatasetSize.value = value

    def getValidationSetFractionalSize(self) -> float:
        return float(self._settings.validationSetFractionalSize.value)

    def setValidationSetFractionalSize(self, value: float) -> None:
        self._settings.validationSetFractionalSize.value = value

    def getOptimizationEpochsPerHalfCycle(self) -> int:
        return self._settings.optimizationEpochsPerHalfCycle.value

    def setOptimizationEpochsPerHalfCycle(self, value: int) -> None:
        self._settings.optimizationEpochsPerHalfCycle.value = value

    def getMaximumLearningRate(self) -> float:
        return float(self._settings.maximumLearningRate.value)

    def setMaximumLearningRate(self, value: float) -> None:
        self._settings.maximumLearningRate.value = value

    # ... methods for other settings continue ...

class PtychoPINNReconstructorLibrary(ReconstructorLibrary, Observer):

    def __init__(self, modelSettings: PtychoPINNModelSettings,
                 trainingSettings: PtychoPINNTrainingSettings,
                 reconstructors: Sequence[Reconstructor]) -> None:
        super().__init__()
        self._modelSettings = modelSettings
        self._trainingSettings = trainingSettings
        self.modelPresenter = PtychoPINNModelPresenter(modelSettings)
        self.trainingPresenter = PtychoPINNTrainingPresenter(trainingSettings)
        self._reconstructors = reconstructors

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PtychoPINNReconstructorLibrary:
        modelSettings = PtychoPINNModelSettings.createInstance(settingsRegistry)
        trainingSettings = PtychoPINNTrainingSettings.createInstance(settingsRegistry)
        modelSettings.addObserver(cls)
        trainingSettings.addObserver(cls)
        modelSettings.addObserver(self.modelPresenter)
        trainingSettings.addObserver(self.trainingPresenter)
        ptychoPINNReconstructor = PtychoPINNTrainableReconstructor(modelSettings, trainingSettings)
        reconstructors = [ptychoPINNReconstructor]
        return cls(modelSettings, trainingSettings, reconstructors)

    @property
    def name(self) -> str:
        return 'PtychoPINN'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
    def update(self, observable: Observable) -> None:
        # Update internal state based on changes in settings
        # This method will be called when settings change
        pass
        # Update internal state based on changes in settings
        pass
