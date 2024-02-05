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

    # ... additional methods for other settings

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

    # ... additional methods for other settings

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
