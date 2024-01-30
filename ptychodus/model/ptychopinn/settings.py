from __future__ import annotations
from pathlib import Path

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class PtychoPINNModelSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        # Define settings specific to PtychoPINN
        # Example:
        self.learningRate = settingsGroup.createRealEntry('LearningRate', '1e-3')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PtychoPINNModelSettings:
        settingsGroup = settingsRegistry.createGroup('PtychoPINN')
        settings = cls(settingsGroup)
        settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PtychoPINNTrainingSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        # Define training settings specific to PtychoPINN
        # Example:
        self.epochs = settingsGroup.createIntegerEntry('Epochs', 100)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PtychoPINNTrainingSettings:
        settingsGroup = settingsRegistry.createGroup('PtychoPINNTraining')
        settings = cls(settingsGroup)
        settingsGroup.addObserver(settings)
        return settings
