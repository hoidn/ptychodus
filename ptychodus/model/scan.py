from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional
import csv
import logging

import numpy

from .chooser import StrategyChooser, StrategyEntry
from .geometry import Interval, Box
from .observer import Observable, Observer
from .settings import SettingsRegistry, SettingsGroup

logger = logging.getLogger(__name__)


class ScanSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.initializer = settingsGroup.createStringEntry('Initializer', 'Snake')
        self.customFileType = settingsGroup.createStringEntry('CustomFileType', 'CSV')
        self.customFilePath = settingsGroup.createPathEntry('CustomFilePath', None)
        self.extentX = settingsGroup.createIntegerEntry('ExtentX', 10)
        self.extentY = settingsGroup.createIntegerEntry('ExtentY', 10)
        self.stepSizeXInMeters = settingsGroup.createRealEntry('StepSizeXInMeters', '1e-6')
        self.stepSizeYInMeters = settingsGroup.createRealEntry('StepSizeYInMeters', '1e-6')
        self.jitterRadiusInPixels = settingsGroup.createRealEntry('JitterRadiusInPixels', '0')
        self.transform = settingsGroup.createStringEntry('Transform', '+X+Y')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> ScanSettings:
        settings = cls(settingsRegistry.createGroup('Scan'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


@dataclass(frozen=True)
class ScanPoint:
    x: Decimal
    y: Decimal


class CartesianScanInitializer(Sequence[ScanPoint]):
    def __init__(self, settings: ScanSettings, snake: bool) -> None:
        super().__init__()
        self._settings = settings
        self._snake = snake

    @classmethod
    def createRasterInstance(cls, settings: ScanSettings) -> CartesianScanInitializer:
        return cls(settings, False)

    @classmethod
    def createSnakeInstance(cls, settings: ScanSettings) -> CartesianScanInitializer:
        return cls(settings, True)

    def __getitem__(self, index: int) -> ScanPoint:
        if index >= len(self):
            raise IndexError(f'Index {index} is out of range')

        nx = self._settings.extentX.value
        y, x = divmod(index, nx)

        if self._snake and y & 1:
            x = nx - 1 - x

        x *= self._settings.stepSizeXInMeters.value
        y *= self._settings.stepSizeYInMeters.value

        return ScanPoint(x, y)

    def __len__(self) -> int:
        nx = self._settings.extentX.value
        ny = self._settings.extentY.value
        return nx * ny

    def __str__(self) -> str:
        return 'Snake' if self._snake else 'Raster'


class SpiralScanInitializer(Sequence[ScanPoint]):
    def __init__(self, settings: ScanSettings) -> None:
        super().__init__()
        self._settings = settings

    def __getitem__(self, index: int) -> ScanPoint:
        if index >= len(self):
            raise IndexError(f'Index {index} is out of range')

        # theta = omega * t
        # r = a + b * theta
        # x = r * numpy.cos(theta)
        # y = r * numpy.sin(theta)

        sqrtIndex = Decimal(index).sqrt()

        # TODO generalize parameters and redo without casting to float
        theta = float(4 * sqrtIndex)
        cosTheta = Decimal(numpy.cos(theta))
        sinTheta = Decimal(numpy.sin(theta))

        x = sqrtIndex * cosTheta * self._settings.stepSizeXInMeters.value
        y = sqrtIndex * sinTheta * self._settings.stepSizeYInMeters.value

        return ScanPoint(x, y)

    def __len__(self) -> int:
        nx = self._settings.extentX.value
        ny = self._settings.extentY.value
        return nx * ny

    def __str__(self) -> str:
        return 'Spiral'


class ScanFileReader(ABC):
    @abstractmethod
    def getFileFilter(self) -> str:
        pass

    @abstractmethod
    def read(self, filePath: Path) -> Iterable[ScanPoint]:
        pass


class ScanPointParseError(Exception):
    pass


class CSVScanFileReader(ScanFileReader):
    def __init__(self) -> None:
        self._xcol = 1
        self._ycol = 0

    def getFileFilter(self) -> str:
        return 'Comma-Separated Values Files (*.csv)'

    def read(self, filePath: Path) -> Iterable[ScanPoint]:
        scanPointList = list()
        minimumColumnCount = max(self._xcol, self._ycol) + 1

        with open(filePath, newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')

            for row in csvReader:
                if row[0].startswith('#'):
                    continue

                if len(row) < minimumColumnCount:
                    raise ScanPointParseError()

                x = Decimal(row[self._xcol])
                y = Decimal(row[self._ycol])
                point = ScanPoint(x, y)

                scanPointList.append(point)

        return scanPointList

    def __str__(self) -> str:
        return 'CSV'


class CustomScanInitializer(Sequence[ScanPoint], Observer):
    def __init__(self, settings: ScanSettings, fileReaderList: list[ScanFileReader]) -> None:
        super().__init__()
        self._settings = settings
        self._fileReaderList = fileReaderList.copy()
        self._fileReader = fileReaderList[0]
        self._mapFileFilterToFileType = {
            fileReader.getFileFilter(): str(fileReader)
            for fileReader in fileReaderList
        }
        self._pointList: list[ScanPoint] = list()

    @classmethod
    def createInstance(cls, settings: ScanSettings,
                       fileReaderList: list[ScanFileReader]) -> CustomScanInitializer:
        initializer = cls(settings, fileReaderList)
        initializer._updateFileReader()
        initializer._openScanFromSettings()
        settings.customFileType.addObserver(initializer)
        settings.customFilePath.addObserver(initializer)
        return initializer

    def __getitem__(self, index: int) -> ScanPoint:
        return self._pointList[index]

    def __len__(self) -> int:
        return len(self._pointList)

    def __str__(self) -> str:
        return 'Custom'

    def getOpenFileFilterList(self) -> list[str]:
        return [reader.getFileFilter() for reader in self._fileReaderList]

    def _updateFileReader(self) -> None:
        fileType = self._settings.customFileType.value

        try:
            self._fileReader = next(reader for reader in self._fileReaderList
                                    if fileType.casefold() == str(reader).casefold())
        except StopIteration:
            logger.debug(f'Invalid scan file type \"{fileType}\"')
            return

        self._settings.customFileType.value = str(self._fileReader)

    def _openScan(self, filePath: Path) -> None:
        if filePath is not None and filePath.is_file():
            pointIterable = self._fileReader.read(filePath)
            self._pointList = [point for point in pointIterable]

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        try:
            fileType = self._mapFileFilterToFileType[fileFilter]
            self._settings.customFileType.value = fileType
        except KeyError:
            logger.debug(f'Invalid scan file filter \"{fileFilter}\"')
            return

        self._settings.customFilePath.value = filePath

    def _openScanFromSettings(self) -> None:
        self._openScan(self._settings.customFilePath.value)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.customFileType:
            self._updateFileReader()
        elif observable is self._settings.customFilePath:
            self._openScanFromSettings()


class ScanPointTransform(Enum):
    PXPY = 0x0
    MXPY = 0x1
    PXMY = 0x2
    MXMY = 0x3
    PYPX = 0x4
    PYMX = 0x5
    MYPX = 0x6
    MYMX = 0x7

    @property
    def negateX(self) -> bool:
        return self.value & 1 != 0

    @property
    def negateY(self) -> bool:
        return self.value & 2 != 0

    @property
    def swapXY(self) -> bool:
        return self.value & 4 != 0

    @property
    def simpleName(self) -> str:
        xp = '-x' if self.negateX else '+x'
        yp = '-y' if self.negateY else '+y'
        return f'{yp}{xp}' if self.swapXY else f'{xp}{yp}'

    @property
    def displayName(self) -> str:
        xp = '\u2212x' if self.negateX else '\u002Bx'
        yp = '\u2212y' if self.negateY else '\u002By'
        return f'({yp}, {xp})' if self.swapXY else f'({xp}, {yp})'

    def __call__(self, point: ScanPoint) -> ScanPoint:
        xp = -point.x if self.negateX else point.x
        yp = -point.y if self.negateY else point.y
        return ScanPoint(yp, xp) if self.swapXY else ScanPoint(xp, yp)


class Scan(Sequence[ScanPoint], Observable, Observer):
    @staticmethod
    def _createTransformEntry(xform: ScanPointTransform) -> StrategyEntry[ScanPointTransform]:
        return StrategyEntry[ScanPointTransform](simpleName=xform.simpleName,
                                                 displayName=xform.displayName,
                                                 strategy=xform)

    def __init__(self, settings: ScanSettings) -> None:
        super().__init__()
        self._settings = settings
        self._scanPointList: list[ScanPoint] = list()
        self._boundingBoxInMeters: Optional[Box[Decimal]] = None
        self._transformChooser = StrategyChooser[ScanPointTransform].createFromList(
            [Scan._createTransformEntry(xform) for xform in ScanPointTransform])

    @classmethod
    def createInstance(cls, settings: ScanSettings) -> Scan:
        scan = cls(settings)
        settings.transform.addObserver(scan)
        scan._transformChooser.addObserver(scan)
        scan._syncTransformFromSettings()
        return scan

    def __getitem__(self, index: int) -> ScanPoint:
        transform = self._transformChooser.getCurrentStrategy()
        return transform(self._scanPointList[index])

    def __len__(self) -> int:
        return len(self._scanPointList)

    def setScanPoints(self, scanPointIterable: Iterable[ScanPoint]) -> None:
        self._scanPointList = [scanPoint for scanPoint in scanPointIterable]
        self._updateBoundingBox()
        self.notifyObservers()

    def getSaveFileFilter() -> str:
        return 'Comma-Separated Values Files (*.csv)'

    def write(self, filePath: Path) -> None:
        with open(filePath, 'wt') as csvFile:
            for point in self._scanPointList:
                csvFile.write(f'{point.y},{point.x}\n')

    def getBoundingBoxInMeters(self) -> Optional[Box[Decimal]]:
        return self._boundingBoxInMeters

    def _updateBoundingBox(self) -> None:
        boundingBoxInMeters = None
        scanPointIterator = iter(self)

        try:
            scanPoint = next(scanPointIterator)
            xint = Interval(scanPoint.x, scanPoint.x)
            yint = Interval(scanPoint.y, scanPoint.y)
            boundingBoxInMeters = Box[Decimal]((xint, yint))

            while True:
                scanPoint = next(scanPointIterator)
                boundingBoxInMeters[0].hull(scanPoint.x)
                boundingBoxInMeters[1].hull(scanPoint.y)
        except StopIteration:
            pass

        self._boundingBoxInMeters = boundingBoxInMeters

    def getTransform(self) -> str:
        return self._transformChooser.getCurrentDisplayName()

    def setTransform(self, name: str) -> None:
        self._transformChooser.setFromDisplayName(name)

    def _syncTransformFromSettings(self) -> None:
        self._transformChooser.setFromSimpleName(self._settings.transform.value)

    def _syncTransformToSettings(self) -> None:
        self._updateBoundingBox()
        self._settings.transform.value = self._transformChooser.getCurrentSimpleName()
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._settings.transform:
            self._syncTransformFromSettings()
        elif observable is self._transformChooser:
            self._syncTransformToSettings()


class ScanInitializer(Observable, Observer):
    def __init__(self, settings: ScanSettings, scan: Scan,
                 customInitializer: CustomScanInitializer, reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._scan = scan
        self._reinitObservable = reinitObservable
        self._customInitializer = customInitializer
        self._initializer = customInitializer
        self._initializerList: list[Sequence[ScanPoint]] = [customInitializer]

    @classmethod
    def createInstance(cls, settings: ScanSettings, scan: Scan,
                       customInitializer: CustomScanInitializer,
                       reinitObservable: Observable) -> ScanInitializer:
        initializer = cls(settings, scan, customInitializer, reinitObservable)
        initializer.addInitializer(SpiralScanInitializer(settings))
        initializer.addInitializer(CartesianScanInitializer.createSnakeInstance(settings))
        initializer.addInitializer(CartesianScanInitializer.createRasterInstance(settings))
        initializer.setInitializerFromSettings()
        settings.initializer.addObserver(initializer)
        reinitObservable.addObserver(initializer)
        return initializer

    def addInitializer(self, initializer: Sequence[ScanPoint]) -> None:
        self._initializerList.insert(0, initializer)

    def getInitializerList(self) -> list[str]:
        return [str(initializer) for initializer in self._initializerList]

    def getInitializer(self) -> str:
        return str(self._initializer)

    def _setInitializer(self, initializer: Sequence[ScanPoint]) -> None:
        if initializer is not self._initializer:
            self._initializer = initializer
            self._settings.initializer.value = str(self._initializer)
            self.notifyObservers()

    def setInitializer(self, name: str) -> None:
        try:
            initializer = next(ini for ini in self._initializerList
                               if name.casefold() == str(ini).casefold())
        except StopIteration:
            logger.debug(f'Invalid initializer \"{name}\"')
            return

        self._setInitializer(initializer)

    def setInitializerFromSettings(self) -> None:
        self.setInitializer(self._settings.initializer.value)

    def initializeScan(self) -> None:
        self._scan.setScanPoints(self._initializer)

    def getOpenFileFilterList(self) -> list[str]:
        return self._customInitializer.getOpenFileFilterList()

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        self._customInitializer.openScan(filePath, fileFilter)
        self._setInitializer(self._customInitializer)
        self.initializeScan()

    def getSaveFileFilterList(self) -> list[str]:
        return [self._scan.getSaveFileFilter()]

    def saveScan(self, filePath: Path, fileFilter: str) -> None:
        self._scan.write(filePath)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.initializer:
            self.setInitializerFromSettings()
        elif observable is self._reinitObservable:
            self.initializeScan()


class ScanPresenter(Observable, Observer):
    MAX_INT = 0x7FFFFFFF

    def __init__(self, settings: ScanSettings, scan: Scan, initializer: ScanInitializer) -> None:
        super().__init__()
        self._settings = settings
        self._scan = scan
        self._initializer = initializer

    @classmethod
    def createInstance(cls, settings: ScanSettings, scan: Scan,
                       initializer: ScanInitializer) -> ScanPresenter:
        presenter = cls(settings, scan, initializer)
        settings.addObserver(presenter)
        scan.addObserver(presenter)
        initializer.addObserver(presenter)
        return presenter

    def getOpenFileFilterList(self) -> list[str]:
        return self._initializer.getOpenFileFilterList()

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        self._initializer.openScan(filePath, fileFilter)

    def getSaveFileFilterList(self) -> list[str]:
        return self._initializer.getSaveFileFilterList()

    def saveScan(self, filePath: Path, fileFilter: str) -> None:
        self._initializer.saveScan(filePath, fileFilter)

    def getInitializerList(self) -> list[str]:
        return self._initializer.getInitializerList()

    def getInitializer(self) -> str:
        return self._initializer.getInitializer()

    def setInitializer(self, name: str) -> None:
        self._initializer.setInitializer(name)

    def initializeScan(self) -> None:
        self._initializer.initializeScan()

    def getTransformList(self) -> list[str]:
        return [str(xform) for xform in ScanPointTransform]

    def getTransform(self) -> str:
        return self._scan.getTransform()

    def setTransform(self, name: str) -> None:
        self._scan.setTransform(name)

    def getScanPointList(self) -> list[ScanPoint]:
        return [scanPoint for scanPoint in self._scan]

    def getNumberOfScanPointsLimits(self) -> Interval[int]:
        return Interval[int](0, self.MAX_INT)

    def getNumberOfScanPoints(self) -> int:
        limits = self.getNumberOfScanPointsLimits()
        return limits.clamp(len(self._scan))

    def getExtentXLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getExtentX(self) -> int:
        limits = self.getExtentXLimits()
        return limits.clamp(self._settings.extentX.value)

    def setExtentX(self, value: int) -> None:
        self._settings.extentX.value = value

    def getExtentYLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getExtentY(self) -> int:
        limits = self.getExtentYLimits()
        return limits.clamp(self._settings.extentY.value)

    def setExtentY(self, value: int) -> None:
        self._settings.extentY.value = value

    def getStepSizeXInMeters(self) -> Decimal:
        return self._settings.stepSizeXInMeters.value

    def setStepSizeXInMeters(self, value: Decimal) -> None:
        self._settings.stepSizeXInMeters.value = value

    def getStepSizeYInMeters(self) -> Decimal:
        return self._settings.stepSizeYInMeters.value

    def setStepSizeYInMeters(self, value: Decimal) -> None:
        self._settings.stepSizeYInMeters.value = value

    def getJitterRadiusInPixels(self) -> Decimal:
        return self._settings.jitterRadiusInPixels.value

    def setJitterRadiusInPixels(self, value: Decimal) -> None:
        self._settings.jitterRadiusInPixels.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._scan:
            self.notifyObservers()
        elif observable is self._initializer:
            self.notifyObservers()
