from __future__ import annotations
from collections.abc import Iterator, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Final
import logging

from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.reconstructor import (NullReconstructor, Reconstructor, ReconstructorLibrary,
                                  TrainableReconstructor)
from ...api.settings import SettingsRegistry
from .settings import PtychoPINNModelSettings, PtychoPINNTrainingSettings

logger = logging.getLogger(__name__)


class PtychoPINNModelPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: PtychoPINNModelSettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: PtychoPINNModelSettings) -> PtychoPINNModelPresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    # Define methods to interact with model settings, similar to PtychoNNModelPresenter

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class PtychoPINNTrainingPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: PtychoPINNTrainingSettings) -> None:
        super().__init__()
        self._settings = settings

    def getEpochsLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getEpochs(self) -> int:
        limits = self.getEpochsLimits()
        return limits.clamp(self._settings.epochs.value)

    def setEpochs(self, value: int) -> None:
        self._settings.epochs.value = value

    @classmethod
    def createInstance(cls, settings: PtychoPINNTrainingSettings) -> PtychoPINNTrainingPresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    # Methods to interact with training settings have been defined

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class PtychoPINNReconstructorLibrary(ReconstructorLibrary):

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PtychoPINNReconstructorLibrary:
        modelSettings = PtychoPINNModelSettings.createInstance(settingsRegistry)
        trainingSettings = PtychoPINNTrainingSettings.createInstance(settingsRegistry)
        # Placeholder for adding PtychoPINN specific reconstructors
        ptychoPINNReconstructor: TrainableReconstructor = NullReconstructor('PtychoPINN')
        reconstructors: list[TrainableReconstructor] = [ptychoPINNReconstructor]

        return cls(modelSettings, trainingSettings, reconstructors)

    @property
    def name(self) -> str:
        return 'PtychoPINN'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
