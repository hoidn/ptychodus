from collections.abc import Iterator, Sequence
from ...api.reconstructor import Reconstructor, ReconstructorLibrary
from ...api.settings import SettingsRegistry
from .settings import PtychoPINNModelSettings, PtychoPINNTrainingSettings
from ...api.observer import Observer
from .reconstructor import PtychoPINNTrainableReconstructor

class PtychoPINNReconstructorLibrary(ReconstructorLibrary, Observer):

    def __init__(self, modelSettings: PtychoPINNModelSettings,
                 trainingSettings: PtychoPINNTrainingSettings,
                 reconstructors: Sequence[Reconstructor]) -> None:
        super().__init__()
        self._modelSettings = modelSettings
        self._trainingSettings = trainingSettings
        self._reconstructors = reconstructors

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PtychoPINNReconstructorLibrary:
        modelSettings = PtychoPINNModelSettings.createInstance(settingsRegistry)
        trainingSettings = PtychoPINNTrainingSettings.createInstance(settingsRegistry)
        modelSettings.addObserver(cls)
        trainingSettings.addObserver(cls)
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
        pass
