from pathlib import Path

import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeArrayType, ProbeFileReader, ProbeFileWriter


class CSVProbeFileReader(ProbeFileReader):

    def read(self, filePath: Path) -> ProbeArrayType:
        arrayFlat = numpy.genfromtxt(filePath, delimiter=',', dtype='complex')
        numberOfModes, remainder = divmod(arrayFlat.shape[0], arrayFlat.shape[1])

        if remainder != 0:
            raise ValueError('Failed to determine probe modes!')

        if numberOfModes > 1:
            array = arrayFlat.reshape(numberOfModes, arrayFlat.shape[1], arrayFlat.shape[1])

        return array


class CSVProbeFileWriter(ProbeFileWriter):

    def write(self, filePath: Path, array: ProbeArrayType) -> None:
        arrayFlat = array.reshape(-1, array.shape[-1])
        numpy.savetxt(filePath, arrayFlat, delimiter=',')


def registerPlugins(registry: PluginRegistry) -> None:
    registry.probeFileReaders.registerPlugin(
        CSVProbeFileReader(),
        simpleName='CSV',
        displayName='Comma-Separated Values Files (*.csv)',
    )
    registry.probeFileWriters.registerPlugin(
        CSVProbeFileWriter(),
        simpleName='CSV',
        displayName='Comma-Separated Values Files (*.csv)',
    )
