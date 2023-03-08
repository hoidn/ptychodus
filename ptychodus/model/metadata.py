from __future__ import annotations
from pathlib import Path

import numpy

from ..api.data import DiffractionDataset, DiffractionMetadata
from ..api.observer import Observable, Observer
from .data import DiffractionPatternSettings
from .detector import DetectorSettings
from .probe import ProbeSettings
from .scan import ScanCore


class MetadataPresenter(Observable, Observer):

    def __init__(self, diffractionDataset: DiffractionDataset, detectorSettings: DetectorSettings,
                 patternSettings: DiffractionPatternSettings, probeSettings: ProbeSettings,
                 scanCore: ScanCore) -> None:
        super().__init__()
        self._diffractionDataset = diffractionDataset
        self._detectorSettings = detectorSettings
        self._patternSettings = patternSettings
        self._probeSettings = probeSettings
        self._scanCore = scanCore

    @classmethod
    def createInstance(cls, diffractionDataset: DiffractionDataset,
                       detectorSettings: DetectorSettings,
                       patternSettings: DiffractionPatternSettings, probeSettings: ProbeSettings,
                       scanCore: ScanCore) -> MetadataPresenter:
        presenter = cls(diffractionDataset, detectorSettings, patternSettings, probeSettings,
                        scanCore)
        diffractionDataset.addObserver(presenter)
        return presenter

    @property
    def _metadata(self) -> DiffractionMetadata:
        return self._diffractionDataset.getMetadata()

    def canSyncDetectorPixelCount(self) -> bool:
        return (self._metadata.detectorNumberOfPixels is not None)

    def syncDetectorPixelCount(self) -> None:
        if self._metadata.detectorNumberOfPixels:
            self._detectorSettings.numberOfPixelsX.value = \
                self._metadata.detectorNumberOfPixels.x
            self._detectorSettings.numberOfPixelsY.value = \
                self._metadata.detectorNumberOfPixels.y

    def canSyncDetectorPixelSize(self) -> bool:
        return (self._metadata.detectorPixelSizeInMeters is not None)

    def syncDetectorPixelSize(self) -> None:
        if self._metadata.detectorPixelSizeInMeters:
            self._detectorSettings.pixelSizeXInMeters.value = \
                self._metadata.detectorPixelSizeInMeters.x
            self._detectorSettings.pixelSizeYInMeters.value = \
                self._metadata.detectorPixelSizeInMeters.y

    def canSyncDetectorDistance(self) -> bool:
        return (self._metadata.detectorDistanceInMeters is not None)

    def syncDetectorDistance(self) -> None:
        if self._metadata.detectorDistanceInMeters:
            self._detectorSettings.detectorDistanceInMeters.value = \
                self._metadata.detectorDistanceInMeters

    def canSyncPatternCropCenter(self) -> bool:
        return (self._metadata.cropCenterInPixels is not None \
                or self._metadata.detectorNumberOfPixels is not None)

    def canSyncPatternCropExtent(self) -> bool:
        return (self._metadata.detectorNumberOfPixels is not None)

    def syncPatternCrop(self, syncCenter: bool, syncExtent: bool) -> None:
        if syncCenter:
            if self._metadata.cropCenterInPixels:
                self._patternSettings.cropCenterXInPixels.value = \
                        self._metadata.cropCenterInPixels.x
                self._patternSettings.cropCenterYInPixels.value = \
                        self._metadata.cropCenterInPixels.y
            elif self._metadata.detectorNumberOfPixels:
                self._patternSettings.cropCenterXInPixels.value = \
                        int(self._metadata.detectorNumberOfPixels.x) // 2
                self._patternSettings.cropCenterYInPixels.value = \
                        int(self._metadata.detectorNumberOfPixels.y) // 2

        if syncExtent and self._metadata.detectorNumberOfPixels:
            centerX = self._patternSettings.cropCenterXInPixels.value
            centerY = self._patternSettings.cropCenterYInPixels.value

            extentX = int(self._metadata.detectorNumberOfPixels.x)
            extentY = int(self._metadata.detectorNumberOfPixels.y)

            maxRadiusX = min(centerX, extentX - centerX)
            maxRadiusY = min(centerY, extentY - centerY)
            maxRadius = min(maxRadiusX, maxRadiusY)
            cropDiameterInPixels = 1

            while cropDiameterInPixels < maxRadius:
                cropDiameterInPixels <<= 1

            self._patternSettings.cropExtentXInPixels.value = cropDiameterInPixels
            self._patternSettings.cropExtentYInPixels.value = cropDiameterInPixels

    def canSyncProbeEnergy(self) -> bool:
        return (self._metadata.probeEnergyInElectronVolts is not None)

    def syncProbeEnergy(self) -> None:
        if self._metadata.probeEnergyInElectronVolts:
            self._probeSettings.probeEnergyInElectronVolts.value = \
                    self._metadata.probeEnergyInElectronVolts

    def loadScanFile(self) -> None:  # TODO velociprobe only
        filePathMaster = self._metadata.filePath

        if filePathMaster is None:
            return

        fileName = filePathMaster.stem.replace('master', 'pos') + '.csv'
        filePath = filePathMaster.parents[2] / 'positions' / fileName
        fileFilter = 'Comma-Separated Values Files (*.csv)'  # TODO refactor; get from somewhere
        self._scanCore.openScan(filePath, fileFilter)

    def update(self, observable: Observable) -> None:
        if observable is self._diffractionDataset:
            self.notifyObservers()