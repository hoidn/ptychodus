from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import overload, Any, Optional, Union
import logging

import h5py
import numpy

from ..h5DiffractionFile import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder
from .velociprobeScanFile import VelociprobeScanFileReader
from ptychodus.api.data import (DiffractionDataset, DiffractionFileReader, DiffractionMetadata,
                                DiffractionPatternArray, DiffractionPatternData,
                                DiffractionPatternState, SimpleDiffractionDataset)
from ptychodus.api.geometry import Vector2D
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataGroup:
    arrayList: list[DiffractionPatternArray] = field(default_factory=list)

    @classmethod
    def read(cls, group: h5py.Group) -> DataGroup:
        arrayList: list[DiffractionPatternArray] = list()
        masterFilePath = Path(group.file.filename)

        for name, h5Item in group.items():
            h5Item = group.get(name, getlink=True)

            if isinstance(h5Item, h5py.ExternalLink):
                filePath = masterFilePath.parent / h5Item.filename
                dataPath = str(h5Item.path)
                # TODO use entry/data/data/image_nr_{low,high}
                array = H5DiffractionPatternArray(name, len(arrayList), filePath, dataPath)
                arrayList.append(array)

        return cls(arrayList)

    def __iter__(self) -> Iterator[DiffractionPatternArray]:
        return iter(self.arrayList)

    @overload
    def __getitem__(self, index: int) -> DiffractionPatternArray:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionPatternArray]:
        ...

    def __getitem__(
        self,
        index: Union[int,
                     slice]) -> Union[DiffractionPatternArray, Sequence[DiffractionPatternArray]]:
        return self.arrayList[index]

    def __len__(self) -> int:
        return len(self.arrayList)


@dataclass(frozen=True)
class DetectorSpecificGroup:
    nimages: int
    photon_energy_eV: float
    x_pixels_in_detector: int
    y_pixels_in_detector: int

    @classmethod
    def read(cls, group: h5py.Group) -> DetectorSpecificGroup:
        nimages = group['nimages']
        photonEnergy = group['photon_energy']
        assert photonEnergy.attrs['units'] == b'eV'
        xPixelsInDetector = group['x_pixels_in_detector']
        yPixelsInDetector = group['y_pixels_in_detector']
        return cls(nimages[()], photonEnergy[()], xPixelsInDetector[()], yPixelsInDetector[()])


@dataclass(frozen=True)
class DetectorGroup:
    detectorSpecific: DetectorSpecificGroup
    detector_distance_m: float
    beam_center_x_px: int
    beam_center_y_px: int
    bit_depth_image: int
    x_pixel_size_m: float
    y_pixel_size_m: float

    @classmethod
    def read(cls, group: h5py.Group) -> DetectorGroup:
        detectorSpecific = DetectorSpecificGroup.read(group['detectorSpecific'])
        h5DetectorDistance = group['detector_distance']
        assert h5DetectorDistance.attrs['units'] == b'm'
        h5BeamCenterX = group['beam_center_x']
        assert h5BeamCenterX.attrs['units'] == b'pixel'
        h5BeamCenterY = group['beam_center_y']
        assert h5BeamCenterY.attrs['units'] == b'pixel'
        h5BitDepthImage = group['bit_depth_image']
        h5XPixelSize = group['x_pixel_size']
        assert h5XPixelSize.attrs['units'] == b'm'
        h5YPixelSize = group['y_pixel_size']
        assert h5YPixelSize.attrs['units'] == b'm'
        return cls(detectorSpecific, h5DetectorDistance[()], h5BeamCenterX[()], h5BeamCenterY[()],
                   h5BitDepthImage[()], h5XPixelSize[()], h5YPixelSize[()])


@dataclass(frozen=True)
class InstrumentGroup:
    detector: DetectorGroup

    @classmethod
    def read(cls, group: h5py.Group) -> InstrumentGroup:
        detector = DetectorGroup.read(group['detector'])
        return cls(detector)


@dataclass(frozen=True)
class GoniometerGroup:
    chi_deg: float

    @classmethod
    def read(cls, group: h5py.Group) -> GoniometerGroup:
        chiDataset = group['chi']
        assert chiDataset.attrs['units'] == b'degree'
        chi_deg = float(chiDataset[0])
        return cls(chi_deg)


@dataclass(frozen=True)
class SampleGroup:
    goniometer: GoniometerGroup

    @classmethod
    def read(cls, group: h5py.Group) -> SampleGroup:
        goniometer = GoniometerGroup.read(group['goniometer'])
        return cls(goniometer)


@dataclass(frozen=True)
class EntryGroup:
    data: DataGroup
    instrument: InstrumentGroup
    sample: SampleGroup

    @classmethod
    def read(cls, group: h5py.Group) -> EntryGroup:
        data = DataGroup.read(group['data'])
        instrument = InstrumentGroup.read(group['instrument'])
        sample = SampleGroup.read(group['sample'])
        return cls(data, instrument, sample)


class NeXusDiffractionDataset(DiffractionDataset):

    def __init__(self, metadata: DiffractionMetadata, contentsTree: SimpleTreeNode,
                 entry: EntryGroup) -> None:
        self._metadata = metadata
        self._contentsTree = contentsTree
        self._entry = entry

    def getMetadata(self) -> DiffractionMetadata:
        return self._metadata

    def getContentsTree(self) -> SimpleTreeNode:
        return self._contentsTree

    @overload
    def __getitem__(self, index: int) -> DiffractionPatternArray:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionPatternArray]:
        ...

    def __getitem__(
        self,
        index: Union[int,
                     slice]) -> Union[DiffractionPatternArray, Sequence[DiffractionPatternArray]]:
        return self._entry.data[index]

    def __len__(self) -> int:
        return len(self._entry.data)


class NeXusDiffractionFileReader(DiffractionFileReader):

    def __init__(self, velociprobeScanFileReader: VelociprobeScanFileReader) -> None:
        super().__init__()
        self._velociprobeScanFileReader = velociprobeScanFileReader
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    @property
    def simpleName(self) -> str:
        return 'NeXus'

    @property
    def fileFilter(self) -> str:
        return 'NeXus Master Files (*.h5 *.hdf5)'

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset: DiffractionDataset = SimpleDiffractionDataset.createNullInstance(filePath)

        try:
            with h5py.File(filePath, 'r') as h5File:
                contentsTree = self._treeBuilder.build(h5File)

                try:
                    entry = EntryGroup.read(h5File['entry'])
                    h5Dataset = h5File['/entry/data/data_000001']
                except KeyError:
                    logger.info(f'File {filePath} is not a NeXus data file.')
                else:
                    detector = entry.instrument.detector
                    detectorPixelSizeInMeters = Vector2D[Decimal](
                        Decimal(repr(detector.x_pixel_size_m)),
                        Decimal(repr(detector.y_pixel_size_m)),
                    )
                    cropCenterInPixels = Vector2D[int](
                        int(round(detector.beam_center_x_px)),
                        int(round(detector.beam_center_y_px)),
                    )

                    detectorSpecific = detector.detectorSpecific
                    detectorNumberOfPixels = Vector2D[int](
                        int(detectorSpecific.x_pixels_in_detector),
                        int(detectorSpecific.y_pixels_in_detector),
                    )

                    metadata = DiffractionMetadata(
                        numberOfPatternsPerArray=h5Dataset.shape[0],
                        numberOfPatternsTotal=detectorSpecific.nimages,
                        patternDataType=h5Dataset.dtype,
                        detectorDistanceInMeters=Decimal(repr(detector.detector_distance_m)),
                        detectorNumberOfPixels=detectorNumberOfPixels,
                        detectorPixelSizeInMeters=detectorPixelSizeInMeters,
                        cropCenterInPixels=cropCenterInPixels,
                        probeEnergyInElectronVolts=Decimal(repr(
                            detectorSpecific.photon_energy_eV)),
                        filePath=filePath,
                    )
                    dataset = NeXusDiffractionDataset(metadata, contentsTree, entry)

                    # vvv TODO This is a hack; remove when able! vvv
                    self._velociprobeScanFileReader.setStageRotationInDegrees(
                        entry.sample.goniometer.chi_deg)
        except OSError:
            logger.debug(f'Unable to read file \"{filePath}\".')

        return dataset