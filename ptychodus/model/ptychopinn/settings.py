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
        # Importing settings from params.py
        self.N = settingsGroup.createIntegerEntry('N', 64)
        self.offset = settingsGroup.createIntegerEntry('Offset', 4)
        self.gridsize = settingsGroup.createIntegerEntry('Gridsize', 2)
        self.outer_offset_train = settingsGroup.createIntegerEntry('OuterOffsetTrain', None)
        self.outer_offset_test = settingsGroup.createIntegerEntry('OuterOffsetTest', None)
        self.batch_size = settingsGroup.createIntegerEntry('BatchSize', 16)
        self.nepochs = settingsGroup.createIntegerEntry('NEpochs', 60)
        self.n_filters_scale = settingsGroup.createIntegerEntry('NFiltersScale', 2)
        self.output_prefix = settingsGroup.createStringEntry('OutputPrefix', 'outputs')
        self.big_gridsize = settingsGroup.createIntegerEntry('BigGridsize', 10)
        self.max_position_jitter = settingsGroup.createIntegerEntry('MaxPositionJitter', 10)
        self.sim_jitter_scale = settingsGroup.createRealEntry('SimJitterScale', '0.')
        self.default_probe_scale = settingsGroup.createRealEntry('DefaultProbeScale', '0.7')
        self.mae_weight = settingsGroup.createRealEntry('MAEWeight', '0.')
        self.nll_weight = settingsGroup.createRealEntry('NLLWeight', '1.')
        self.tv_weight = settingsGroup.createRealEntry('TVWeight', '0.')
        self.realspace_mae_weight = settingsGroup.createRealEntry('RealspaceMAEWeight', '0.')
        self.realspace_weight = settingsGroup.createRealEntry('RealspaceWeight', '0.')
        self.nphotons = settingsGroup.createRealEntry('NPhotons', '1e9')
        self.nimgs_train = settingsGroup.createIntegerEntry('NImgsTrain', 9)
        self.nimgs_test = settingsGroup.createIntegerEntry('NImgsTest', 3)
        self.data_source = settingsGroup.createStringEntry('DataSource', 'lines')
        self.probe_trainable = settingsGroup.createBooleanEntry('ProbeTrainable', False)
        self.intensity_scale_trainable = settingsGroup.createBooleanEntry('IntensityScaleTrainable', False)
        self.positions_provided = settingsGroup.createBooleanEntry('PositionsProvided', False)
        self.object_big = settingsGroup.createBooleanEntry('ObjectBig', True)
        self.probe_big = settingsGroup.createBooleanEntry('ProbeBig', False)
        self.probe_scale = settingsGroup.createRealEntry('ProbeScale', '10.')
        self.set_phi = settingsGroup.createBooleanEntry('SetPhi', False)
        self.probe_mask = settingsGroup.createBooleanEntry('ProbeMask', True)
        self.model_type = settingsGroup.createStringEntry('ModelType', 'pinn')
        self.label = settingsGroup.createStringEntry('Label', '')
        self.size = settingsGroup.createIntegerEntry('Size', 392)
        self.amp_activation = settingsGroup.createStringEntry('AmpActivation', 'sigmoid')

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
