from __future__ import annotations
from decimal import Decimal
import logging

try:
    import tike
except ImportError:
    tike = None

from .observer import Observable, Observer
from .reconstructor import Reconstructor
from .settings import SettingsRegistry, SettingsGroup

logger = logging.getLogger(__name__)


class TikeAdaptiveMomentSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.useAdaptiveMoment = settingsGroup.createBooleanEntry('UseAdaptiveMoment', False)
        self.mdecay = settingsGroup.createRealEntry('MDecay', '0.9')
        self.vdecay = settingsGroup.createRealEntry('VDecay', '0.999')
        settingsGroup.addObserver(self)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class TikeAdaptiveMomentPresenter(Observable, Observer):
    def __init__(self, settings: TikeAdaptiveMomentSettings) -> None:
        super().__init__()
        self._settings = settings
        settings.addObserver(self)

    def isAdaptiveMomentEnabled(self) -> bool:
        return self._settings.useAdaptiveMoment.value

    def setAdaptiveMomentEnabled(self, enabled: bool) -> None:
        self._settings.useAdaptiveMoment.value = enabled

    def getMinMDecay(self) -> Decimal:
        return Decimal(0)

    def getMaxMDecay(self) -> Decimal:
        return Decimal(1)

    def getMDecay(self) -> Decimal:
        return self._clamp(self._settings.mdecay.value, self.getMinMDecay(), self.getMaxMDecay())

    def setMDecay(self, value: Decimal) -> None:
        self._settings.mdecay.value = value

    def getMinVDecay(self) -> Decimal:
        return Decimal(0)

    def getMaxVDecay(self) -> Decimal:
        return Decimal(1)

    def getVDecay(self) -> Decimal:
        return self._clamp(self._settings.vdecay.value, self.getMinVDecay(), self.getMaxVDecay())

    def setVDecay(self, value: Decimal) -> None:
        self._settings.vdecay.value = value

    @staticmethod
    def _clamp(x, xmin, xmax):
        assert xmin <= xmax
        return max(xmin, min(x, xmax))

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class TikeProbeCorrectionSettings(TikeAdaptiveMomentSettings):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__(settingsGroup)
        self.useProbeCorrection = settingsGroup.createBooleanEntry('UseProbeCorrection', False)
        self.sparsityConstraint = settingsGroup.createRealEntry('SparsityConstraint', 1)
        self.orthogonalityConstraint = settingsGroup.createBooleanEntry(
            'OrthogonalityConstraint', True)
        self.centeredIntensityConstraint = settingsGroup.createBooleanEntry(
            'CenteredIntensityConstraint', False)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> TikeProbeCorrectionSettings:
        return cls(settingsRegistry.createGroup('TikeProbeCorrection'))


class TikeProbeCorrectionPresenter(TikeAdaptiveMomentPresenter):
    def __init__(self, settings: TikeProbeCorrectionSettings) -> None:
        super().__init__(settings)

    @classmethod
    def createInstance(cls, settings: TikeProbeCorrectionSettings) -> TikeProbeCorrectionPresenter:
        presenter = cls(settings)
        return presenter

    def isProbeCorrectionEnabled(self) -> bool:
        return self._settings.useProbeCorrection.value

    def setProbeCorrectionEnabled(self, enabled: bool) -> None:
        self._settings.useProbeCorrection.value = enabled

    def getMinSparsityConstraint(self) -> Decimal:
        return Decimal(0)

    def getMaxSparsityConstraint(self) -> Decimal:
        return Decimal(1)

    def getSparsityConstraint(self) -> Decimal:
        return self._clamp(self._settings.sparsityConstraint.value,
                           self.getMinSparsityConstraint(), self.getMaxSparsityConstraint())

    def setSparsityConstraint(self, value: Decimal) -> None:
        self._settings.sparsityConstraint.value = value

    def isOrthogonalityConstraintEnabled(self) -> bool:
        return self._settings.orthogonalityConstraint.value

    def setOrthogonalityConstraintEnabled(self, enabled: bool) -> None:
        self._settings.orthogonalityConstraint.value = enabled

    def isCenteredIntensityConstraintEnabled(self) -> bool:
        return self._settings.centeredIntensityConstraint.value

    def setCenteredIntensityConstraintEnabled(self, enabled: bool) -> None:
        self._settings.centeredIntensityConstraint.value = enabled


class TikeObjectCorrectionSettings(TikeAdaptiveMomentSettings):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__(settingsGroup)
        self.useObjectCorrection = settingsGroup.createBooleanEntry('UseObjectCorrection', False)
        self.positivityConstraint = settingsGroup.createRealEntry('PositivityConstraint', 0)
        self.smoothnessConstraint = settingsGroup.createRealEntry('SmoothnessConstraint', 0)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> TikeObjectCorrectionSettings:
        return cls(settingsRegistry.createGroup('TikeObjectCorrection'))


class TikeObjectCorrectionPresenter(TikeAdaptiveMomentPresenter):
    def __init__(self, settings: TikeObjectCorrectionSettings) -> None:
        super().__init__(settings)

    @classmethod
    def createInstance(cls,
                       settings: TikeObjectCorrectionSettings) -> TikeObjectCorrectionPresenter:
        presenter = cls(settings)
        return presenter

    def isObjectCorrectionEnabled(self) -> bool:
        return self._settings.useObjectCorrection.value

    def setObjectCorrectionEnabled(self, enabled: bool) -> None:
        self._settings.useObjectCorrection.value = enabled

    def getMinPositivityConstraint(self) -> Decimal:
        return Decimal(0)

    def getMaxPositivityConstraint(self) -> Decimal:
        return Decimal(1)

    def getPositivityConstraint(self) -> Decimal:
        return self._clamp(self._settings.positivityConstraint.value,
                           self.getMinPositivityConstraint(), self.getMaxPositivityConstraint())

    def setPositivityConstraint(self, value: Decimal) -> None:
        self._settings.positivityConstraint.value = value

    def getMinSmoothnessConstraint(self) -> Decimal:
        return Decimal(0)

    def getMaxSmoothnessConstraint(self) -> Decimal:
        return Decimal('0.125')

    def getSmoothnessConstraint(self) -> Decimal:
        return self._clamp(self._settings.smoothnessConstraint.value,
                           self.getMinSmoothnessConstraint(), self.getMaxSmoothnessConstraint())

    def setSmoothnessConstraint(self, value: Decimal) -> None:
        self._settings.smoothnessConstraint.value = value


class TikePositionCorrectionSettings(TikeAdaptiveMomentSettings):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__(settingsGroup)
        self.usePositionCorrection = settingsGroup.createBooleanEntry(
            'UsePositionCorrection', False)
        self.usePositionRegularization = settingsGroup.createBooleanEntry(
            'UsePositionRegularization', False)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> TikePositionCorrectionSettings:
        return cls(settingsRegistry.createGroup('TikePositionCorrection'))


class TikePositionCorrectionPresenter(TikeAdaptiveMomentPresenter):
    def __init__(self, settings: TikePositionCorrectionSettings) -> None:
        super().__init__(settings)

    @classmethod
    def createInstance(
            cls, settings: TikePositionCorrectionSettings) -> TikePositionCorrectionPresenter:
        presenter = cls(settings)
        return presenter

    def isPositionCorrectionEnabled(self) -> bool:
        return self._settings.usePositionCorrection.value

    def setPositionCorrectionEnabled(self, enabled: bool) -> None:
        self._settings.usePositionCorrection.value = enabled

    def isPositionRegularizationEnabled(self) -> bool:
        return self._settings.usePositionRegularization.value

    def setPositionRegularizationEnabled(self, enabled: bool) -> None:
        self._settings.usePositionRegularization.value = enabled


class TikeSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.useMpi = settingsGroup.createBooleanEntry('UseMpi', False)
        self.numGpus = settingsGroup.createIntegerEntry('NumGpus', 1)
        self.noiseModel = settingsGroup.createStringEntry('NoiseModel', 'gaussian')
        self.numBatch = settingsGroup.createIntegerEntry('NumBatch', 1)
        self.numIter = settingsGroup.createIntegerEntry('NumIter', 1)
        self.cgIter = settingsGroup.createIntegerEntry('CgIter', 2)
        self.alpha = settingsGroup.createRealEntry('Alpha', '0.05')
        self.stepLength = settingsGroup.createRealEntry('StepLength', 1)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> TikeSettings:
        settings = cls(settingsRegistry.createGroup('Tike'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class TikePresenter(Observable, Observer):
    MAX_INT = 0x7FFFFFFF

    def __init__(self, settings: TikeSettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: TikeSettings) -> TikePresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    def isMpiEnabled(self) -> bool:
        return self._settings.useMpi.value

    def setMpiEnabled(self, enabled: bool) -> None:
        self._settings.useMpi.value = enabled

    def getMinNumGpus(self) -> int:
        return 1

    def getMaxNumGpus(self) -> int:
        return self.MAX_INT

    def getNumGpus(self) -> int:
        return self._clamp(self._settings.numGpus.value, self.getMinNumGpus(),
                           self.getMaxNumGpus())

    def setNumGpus(self, value: int) -> None:
        self._settings.numGpus.value = value

    def getNoiseModelList(self) -> list[str]:
        return ['poisson', 'gaussian']

    def getNoiseModel(self) -> str:
        return self._settings.noiseModel.value

    def setNoiseModel(self, name: str) -> None:
        self._settings.noiseModel.value = name

    def getMinNumBatch(self) -> int:
        return 1

    def getMaxNumBatch(self) -> int:
        return self.MAX_INT

    def getNumBatch(self) -> int:
        return self._clamp(self._settings.numBatch.value, self.getMinNumBatch(),
                           self.getMaxNumBatch())

    def setNumBatch(self, value: int) -> None:
        self._settings.numBatch.value = value

    def getMinNumIter(self) -> int:
        return 1

    def getMaxNumIter(self) -> int:
        return self.MAX_INT

    def getNumIter(self) -> int:
        return self._clamp(self._settings.numIter.value, self.getMinNumIter(),
                           self.getMaxNumIter())

    def setNumIter(self, value: int) -> None:
        self._settings.numIter.value = value

    def getMinCgIter(self) -> int:
        return 1

    def getMaxCgIter(self) -> int:
        return 64

    def getCgIter(self) -> int:
        return self._clamp(self._settings.cgIter.value, self.getMinCgIter(), self.getMaxCgIter())

    def setCgIter(self, value: int) -> None:
        self._settings.cgIter.value = value

    def getMinAlpha(self) -> Decimal:
        return Decimal(0)

    def getMaxAlpha(self) -> Decimal:
        return Decimal(1)

    def getAlpha(self) -> Decimal:
        return self._clamp(self._settings.alpha.value, self.getMinAlpha(), self.getMaxAlpha())

    def setAlpha(self, value: Decimal) -> None:
        self._settings.alpha.value = value

    def getMinStepLength(self) -> Decimal:
        return Decimal(0)

    def getMaxStepLength(self) -> Decimal:
        return Decimal(1)

    def getStepLength(self) -> Decimal:
        return self._clamp(self._settings.stepLength.value, self.getMinStepLength(),
                           self.getMaxStepLength())

    def setStepLength(self, value: Decimal) -> None:
        self._settings.stepLength.value = value

    @staticmethod
    def _clamp(x, xmin, xmax):
        assert xmin <= xmax
        return max(xmin, min(x, xmax))

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class TikeReconstructor(Reconstructor):
    """
    data : (FRAME, WIDE, HIGH) float32
        The intensity (square of the absolute value) of the propagated
        wavefront; i.e. what the detector records. FFT-shifted so the
        diffraction peak is at the corners.
    probe : (1, 1, SHARED, WIDE, HIGH) complex64
        The shared complex illumination function amongst all positions.
    scan : (POSI, 2) float32
        Coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi. Coordinate order
        consistent with WIDE, HIGH order.
    algorithm_options : :py:class:`tike.ptycho.solvers.IterativeOptions`
        A class containing algorithm specific parameters
    object_options : :py:class:`tike.ptycho.ObjectOptions`
        A class containing settings related to object updates.
    position_options : :py:class:`tike.ptycho.PositionOptions`
        A class containing settings related to position correction.
    probe_options : :py:class:`tike.ptycho.ProbeOptions`
        A class containing settings related to probe updates.
    psi : (WIDE, HIGH) complex64
        The wavefront modulation coefficients of the object.
    """
    def reconstruct(self, algorithm_options):  # TODO typing
        data = None # FIXME
        probe = None # FIXME
        scan = None # FIXME
        model = None # FIXME
        num_gpu = None # FIXME
        object_options = None # FIXME
        position_options = None # FIXME
        probe_options = None # FIXME
        psi = None # FIXME
        use_mpi = None # FIXME

        result = tike.ptycho.reconstruct(
            data = data,
            probe = probe,
            scan = scan,
            algorithm_options = algorithm_options,
            model = model,
            num_gpu = num_gpu,
            object_options = object_options,
            position_options = position_options,
            probe_options = probe_options,
            psi = initial_object,
            use_mpi = use_mpi,
        )

        return result


class RegularizedPIEReconstructor(Reconstructor):
    @property
    def name(self) -> str:
        return 'rpie'

    @property
    def backendName(self) -> str:
        return 'Tike'

    def reconstruct(self) -> int:
        return 0  # TODO


class AdaptiveMomentGradientDescentReconstructor(Reconstructor):
    @property
    def name(self) -> str:
        return 'adam_grad'

    @property
    def backendName(self) -> str:
        return 'Tike'

    def reconstruct(self) -> int:
        return 0  # TODO


class ConjugateGradientReconstructor(Reconstructor):
    @property
    def name(self) -> str:
        return 'cgrad'

    @property
    def backendName(self) -> str:
        return 'Tike'

    def reconstruct(self) -> int:
        return 0  # TODO


class IterativeLeastSquaresReconstructor(Reconstructor):
    @property
    def name(self) -> str:
        return 'lstsq_grad'

    @property
    def backendName(self) -> str:
        return 'Tike'

    def reconstruct(self) -> int:
        return 0  # TODO


class TikeBackend:
    def __init__(self, settingsRegistry: SettingsRegistry) -> None:
        self._settings = TikeSettings.createInstance(settingsRegistry)
        self._positionCorrectionSettings = TikePositionCorrectionSettings.createInstance(
            settingsRegistry)
        self._probeCorrectionSettings = TikeProbeCorrectionSettings.createInstance(
            settingsRegistry)
        self._objectCorrectionSettings = TikeObjectCorrectionSettings.createInstance(
            settingsRegistry)

        self.presenter = TikePresenter.createInstance(self._settings)
        self.positionCorrectionPresenter = TikePositionCorrectionPresenter.createInstance(
            self._positionCorrectionSettings)
        self.probeCorrectionPresenter = TikeProbeCorrectionPresenter.createInstance(
            self._probeCorrectionSettings)
        self.objectCorrectionPresenter = TikeObjectCorrectionPresenter.createInstance(
            self._objectCorrectionSettings)

        self.reconstructorList: list[Reconstructor] = list()

    @classmethod
    def createInstance(cls,
                       settingsRegistry: SettingsRegistry,
                       isDeveloperModeEnabled: bool = False) -> TikeBackend:
        loadReconstructors = isDeveloperModeEnabled

        if tike:
            logger.info(f'{tike.__name__} ({tike.__version__})')
            loadReconstructors = True
        else:
            logger.info('tike not found.')

        core = cls(settingsRegistry)

        if loadReconstructors:
            core.reconstructorList.append(RegularizedPIEReconstructor())
            core.reconstructorList.append(AdaptiveMomentGradientDescentReconstructor())
            core.reconstructorList.append(ConjugateGradientReconstructor())
            core.reconstructorList.append(IterativeLeastSquaresReconstructor())

        return core
