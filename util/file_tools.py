from abc import ABCMeta, abstractmethod

from typing import Dict, Optional, List
import re
import codecs

import xml.etree.ElementTree as XmlEt

from util.data_tools import DroneProperties, DroneType

from logging import getLogger, NullHandler, StreamHandler, INFO, DEBUG

logger = getLogger(__name__)
logger.addHandler(NullHandler())


# logger.setLevel(DEBUG)  # for standalone debugging
# logger.addHandler(StreamHandler()) # for standalone debugging


class FileHandler(metaclass=ABCMeta):
    def __init__(self, encoding: str = 'utf-8'):
        self._codec = encoding

    @staticmethod
    def suffix_check(suffix_list: List[str], file_name: str) -> bool:
        suffix_list = [s.strip('.') for s in suffix_list]
        suffix = '|'.join(suffix_list).upper()
        pattern_str = r'\.(' + suffix + r')$'
        if not re.search(pattern_str, file_name.upper()):
            mes = f"""
            Suffix of the file name should be '{suffix}'.
            Current file name ： {file_name}
            """
            logger.error(mes)
            return False
        return True

    def read(self, file_name: str):
        try:
            with codecs.open(file_name, 'r', self._codec) as srw:
                return self.read_handling(srw)
        except FileNotFoundError:
            logger.error(f"{file_name} can not be found ...")
        except OSError as e:
            logger.error(f"OS error occurred trying to read {file_name}")
            logger.error(e)

    def write(self, data, file_name: str):
        try:
            with codecs.open(file_name, 'w', self._codec) as srw:
                self.write_handling(data, srw)
        except OSError as e:
            logger.error(f"OS error occurred trying to write {file_name}")
            logger.error(e)

    @abstractmethod
    def read_handling(self, srw: codecs.StreamReaderWriter):
        raise NotImplementedError

    @abstractmethod
    def write_handling(self, data, srw: codecs.StreamReaderWriter):
        raise NotImplementedError


class DroneUrdfAnalyzer(FileHandler):
    def __init__(self, codec: str = 'utf-8'):
        super().__init__(codec)

    def read_handling(self, srw: codecs.StreamReaderWriter):
        return XmlEt.parse(srw)

    def write_handling(self, data, srw: codecs.StreamReaderWriter):
        pass

    def parse(self, urdf_file: str, drone_type: int = 0, g: float = 9.8) -> Optional[DroneProperties]:
        # check the file suffix
        if not self.suffix_check(['.urdf'], urdf_file):
            return None

        et = self.read(urdf_file)

        if et is None:
            return None

        root = et.getroot()
        prop = root[0]
        link = root[1]  # first link -> link name="base_link"

        dataset = DroneProperties(
            type=drone_type,
            g=g,
            m=float(link[0][1].attrib['value']),
            l=float(prop.attrib['arm']),
            thrust2weight_ratio=float(prop.attrib['thrust2weight']),
            ixx=float(link[0][2].attrib['ixx']),
            iyy=float(link[0][2].attrib['iyy']),
            izz=float(link[0][2].attrib['izz']),
            kf=float(prop.attrib['kf']),
            km=float(prop.attrib['km']),
            collision_h=float(link[2][1][0].attrib['length']),
            collision_r=float(link[2][1][0].attrib['radius']),
            collision_shape_offsets=[float(s) for s in link[2][0].attrib['xyz'].split(' ')],
            max_speed_kmh=float(prop.attrib['max_speed_kmh']),
            gnd_eff_coeff=float(prop.attrib['gnd_eff_coeff']),
            prop_radius=float(prop.attrib['prop_radius']),
            drag_coeff_xy=float(prop.attrib['drag_coeff_xy']),
            dw_coeff_1=float(prop.attrib['dw_coeff_1']),
            dw_coeff_2=float(prop.attrib['dw_coeff_2']),
            dw_coeff_3=float(prop.attrib['dw_coeff_3']),
        )

        return dataset


def get_drone_properties(file_path: str, drone_type: DroneType) -> DroneProperties:
    file_analyzer = DroneUrdfAnalyzer()
    return file_analyzer.parse(file_path, int(drone_type))


if __name__ == "__main__":
    file = 'assets/drone_p_01.urdf'
    urdf_a = DroneUrdfAnalyzer()

    parms = urdf_a.parse(file)

    print(parms.drag_coeff)
