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

    def getLearningRateLimits(self) -> Interval[float]:
        return Interval[float](1e-6, 1.0)

    # Add methods to interact with individual settings here
    # Similar to those in PtychoNNModelPresenter
    # Example:
    MAX_INT: Final[int] = 0x7FFFFFFF
    MIN_FLOAT: Final[float] = 1e-38
    MAX_FLOAT: Final[float] = 1e38



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

    def getOutputPrefix(self) -> str:
        return self._settings.output_prefix.value

    def setOutputPrefix(self, value: str) -> None:
        self._settings.output_prefix.value = value

    def getDefaultProbeScale(self) -> float:
        return float(self._settings.default_probe_scale.value)

    def setDefaultProbeScale(self, value: float) -> None:
        self._settings.default_probe_scale.value = value

    def getNPhotons(self) -> float:
        return float(self._settings.nphotons.value)

    def setNPhotons(self, value: float) -> None:
        self._settings.nphotons.value = value

    def getObjectBig(self) -> bool:
        return self._settings.object_big.value

    def setObjectBig(self, value: bool) -> None:
        self._settings.object_big.value = value

    def getProbeBig(self) -> bool:
        return self._settings.probe_big.value

    def setProbeBig(self, value: bool) -> None:
        self._settings.probe_big.value = value

    def getModelType(self) -> str:
        return self._settings.model_type.value

    def setModelType(self, value: str) -> None:
        self._settings.model_type.value = value

    def getAmpActivation(self) -> str:
        return self._settings.amp_activation.value

    def setAmpActivation(self, value: str) -> None:
        self._settings.amp_activation.value = value

    def getProbeTrainable(self) -> bool:
        return self._settings.probe_trainable.value

    def setProbeTrainable(self, value: bool) -> None:
        self._settings.probe_trainable.value = value

    def getIntensityScaleTrainable(self) -> bool:
        return self._settings.intensity_scale_trainable.value

    def setIntensityScaleTrainable(self, value: bool) -> None:
        self._settings.intensity_scale_trainable.value = value

    def getPositionsProvided(self) -> bool:
        return self._settings.positions_provided.value

    def setPositionsProvided(self, value: bool) -> None:
        self._settings.positions_provided.value = value
    def getNFiltersScaleLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def setNFiltersScale(self, value: int) -> None:
        self._settings.n_filters_scale.value = value

    def getProbeScaleLimits(self) -> Interval[float]:
        return Interval[float](0.1, 100.0)

    def getProbeScale(self) -> float:
        return float(self._settings.probe_scale.value)

    def setProbeScale(self, value: float) -> None:
        self._settings.probe_scale.value = value

    def getProbeMask(self) -> bool:
        return self._settings.probe_mask.value

    def setProbeMask(self, value: bool) -> None:
        self._settings.probe_mask.value = value

    def getMaximumTrainingDatasetSizeLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getMaximumTrainingDatasetSize(self) -> int:
        return self._settings.maximumTrainingDatasetSize.value

    def setMaximumTrainingDatasetSize(self, value: int) -> None:
        self._settings.maximumTrainingDatasetSize.value = value

    def getValidationSetFractionalSizeLimits(self) -> Interval[float]:
        return Interval[float](0.0, 1.0)

    def setValidationSetFractionalSize(self, value: float) -> None:
        self._settings.validationSetFractionalSize.value = value

    def getOptimizationEpochsPerHalfCycle(self) -> int:
        return self._settings.optimizationEpochsPerHalfCycle.value

    def setOptimizationEpochsPerHalfCycle(self, value: int) -> None:
        self._settings.optimizationEpochsPerHalfCycle.value = value

    def getMaximumLearningRateLimits(self) -> Interval[float]:
        return Interval[float](self.MIN_FLOAT, self.MAX_FLOAT)

    def setMaximumLearningRate(self, value: float) -> None:
        self._settings.maximumLearningRate.value = value

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
    def getMAEWeightLimits(self) -> Interval[float]:
        return Interval[float](0.0, self.MAX_FLOAT)

    def getMAEWeightLimits(self) -> Interval[float]:
        return Interval[float](0.0, self.MAX_FLOAT)

    def getMAEWeightLimits(self) -> Interval[float]:
        return Interval[float](0.0, 1.0)

    def getMAEWeight(self) -> float:
        return float(self._settings.mae_weight.value)

    def setMAEWeight(self, value: float) -> None:
        self._settings.mae_weight.value = value

    def getNLLWeightLimits(self) -> Interval[float]:
        return Interval[float](0.0, self.MAX_FLOAT)

    def getNLLWeightLimits(self) -> Interval[float]:
        return Interval[float](0.0, self.MAX_FLOAT)

    def getNLLWeightLimits(self) -> Interval[float]:
        return Interval[float](0.0, 1.0)

    def getNLLWeight(self) -> float:
        return float(self._settings.nll_weight.value)

    def setNLLWeight(self, value: float) -> None:
        self._settings.nll_weight.value = value

    def getTVWeightLimits(self) -> Interval[float]:
        return Interval[float](0.0, self.MAX_FLOAT)

    def getTVWeightLimits(self) -> Interval[float]:
        return Interval[float](0.0, self.MAX_FLOAT)

    def getTVWeightLimits(self) -> Interval[float]:
        return Interval[float](0.0, 1.0)

    def getTVWeight(self) -> float:
        return float(self._settings.tv_weight.value)

    def setTVWeight(self, value: float) -> None:
        self._settings.tv_weight.value = value

    def getRealspaceMAEWeightLimits(self) -> Interval[float]:
        return Interval[float](0.0, self.MAX_FLOAT)

    def getRealspaceMAEWeightLimits(self) -> Interval[float]:
        return Interval[float](0.0, self.MAX_FLOAT)

    def getRealspaceMAEWeightLimits(self) -> Interval[float]:
        return Interval[float](0.0, 1.0)

    def getRealspaceMAEWeight(self) -> float:
        return float(self._settings.realspace_mae_weight.value)

    def setRealspaceMAEWeight(self, value: float) -> None:
        self._settings.realspace_mae_weight.value = value

    def getRealspaceWeightLimits(self) -> Interval[float]:
        return Interval[float](0.0, self.MAX_FLOAT)

    def getRealspaceWeightLimits(self) -> Interval[float]:
        return Interval[float](0.0, self.MAX_FLOAT)

    def getRealspaceWeightLimits(self) -> Interval[float]:
        return Interval[float](0.0, 1.0)

    def getRealspaceWeight(self) -> float:
        return float(self._settings.realspace_weight.value)

    def setRealspaceWeight(self, value: float) -> None:
        self._settings.realspace_weight.value = value

    def getSizeLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getSizeLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getSizeLimits(self) -> Interval[int]:
        return Interval[int](64, 4096)

    def getSize(self) -> int:
        return self._settings.size.value

    def setSize(self, value: int) -> None:
        self._settings.size.value = value
