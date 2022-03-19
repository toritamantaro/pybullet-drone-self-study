from typing import Optional, Tuple, Sequence
import math

import numpy as np
import pybullet as p

from dev.device_base import CallableDeviceBase

from logging import getLogger, NullHandler, StreamHandler, INFO, DEBUG, WARNING

logger = getLogger(__name__)
logger.addHandler(NullHandler())


# logger.setLevel(DEBUG)  # for standalone debugging
# logger.addHandler(StreamHandler())  # for standalone debugging

def compute_view_matrix_from_cam_location(
        cam_pos: Sequence[float],
        cam_quat: Sequence[float],
        target_distance: float = 10.,
) -> Sequence[float]:
    """
    Parameters
    ----------
    cam_pos : The position of the camera. It is tuple or list with 3 elements.
    cam_quat : The quaternion of the camera. It is tuple or list with 4 elements
    target_distance : Distance from camera to subject.

    Returns
    -------
    view_matrix: [4x4]=[16] It is flatten tuple or list with 16 elements.
    """
    cam_rot_mat = p.getMatrixFromQuaternion(cam_quat)
    forward_vec = [cam_rot_mat[0], cam_rot_mat[3], cam_rot_mat[6]]
    cam_up_vec = [cam_rot_mat[2], cam_rot_mat[5], cam_rot_mat[8]]
    cam_target = [
        cam_pos[0] + forward_vec[0] * target_distance,
        cam_pos[1] + forward_vec[1] * target_distance,
        cam_pos[2] + forward_vec[2] * target_distance,
    ]
    view_mat = p.computeViewMatrix(cam_pos, cam_target, cam_up_vec)
    return view_mat


def compute_pose_from_view_matrix(view_matrix: Sequence[float]) -> np.ndarray:
    """
    Convert camera view matrix to pose matrix

    Parameters
    ----------
    view_matrix : [4x4]=[16] Flatten list with 16 float elements.

    Returns
    -------
    pose_matrix : [4x4]
    """
    cam_pose_matrix = np.linalg.inv(np.array(view_matrix).reshape(4, 4).T)
    cam_pose_matrix[:, 1:3] = -cam_pose_matrix[:, 1:3]
    return cam_pose_matrix


class BulletCameraDevice(CallableDeviceBase):
    """
    This camera can be used in the PyBullet environment by specifying the viewing angle and clipping range.

    クリッピングの範囲や画角、画素数を定義して、Pybullet内で用いるカメラのカメラ内部パラメータを含む
    カメラの内部行列と射影行列（クリッピング空間内の物体をスクリーンに投影）を保持させる
    """

    def __init__(
            self,
            z_near: float,
            z_far: float,
            res_w: int = 640,
            res_h: int = 480,
            fov_w: float = 50.,
    ):
        """
        Parameters
        ----------
        z_near: value of the distance to near side of clipping space in maters.
        z_far: value of the distance to far side of clipping space in maters.
        res_w: Horizontal camera resolution in pixels.
        res_h: Vertical camera resolution in pixels.
        fov_w: horizontal field of view in degrees.
        """
        self._z_near = z_near
        self._z_far = z_far
        self._width = res_w
        self._height = res_h
        self._fov_width_deg = fov_w
        # This is the focal length in pixels, one of the camara intrinsic parameter.
        self._focal_length_pix = (float(self._width) / 2) / math.tan((math.pi * self._fov_width_deg / 180) / 2)
        self._fov_height_deg = (math.atan((float(self._height) / 2) / self._focal_length_pix) * 2 / math.pi) * 180
        self._intrinsic_matrix = self.compute_intrinsic_matrix()
        self._projection_matrix = self.compute_projection_matrix()

    def get_intrinsic_matrix(self):
        return self._intrinsic_matrix

    def get_projection_matrix(self):
        return self._projection_matrix

    def open(self):
        pass

    def close(self):
        pass

    def __call__(self, view_matrix: Sequence[float]) -> Optional[np.ndarray]:
        return self.cam_capture(view_matrix)

    def compute_projection_matrix(self) -> np.ndarray:
        """
        Compute projection matrix from the attributes of this class by PyBullet function.

        Returns
        -------
        mat: projection matrix [4x4]

        """
        ''' Reference of p.computeProjectionMatrixFOV(()
        This command also will return a 4x4 projection matrix, using different parameters. 
        You can check out OpenGL documentation for the meaning of the parameters.
        '''
        mat = p.computeProjectionMatrixFOV(
            fov=self._fov_height_deg,  # field of view (fovy)
            aspect=float(self._width) / float(self._height),
            nearVal=self._z_near,  # near plane distance
            farVal=self._z_far,  # far plane distance
        )
        return mat

    def compute_intrinsic_matrix(self) -> np.ndarray:
        """
        Compute camera intrinsic matrix from the attributes of this class.
        All of these intrinsic parameters are in pixels.

        Returns
        -------
        mat: camera intrinsic matrix [3x3]
        """
        mat = np.array(
            [[self._focal_length_pix, 0, float(self._width) / 2],
             [0, self._focal_length_pix, float(self._height) / 2],
             [0, 0, 1]]
        )
        return mat

    def cam_capture(self, view_matrix: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Capture an image in PyBullet virtual space by using PyBullet's getCameraImage() function.

        This function requires the view_matrix[4x4]->[16] as an argument.
        The view matrix can be calculated using PyBullet's copbuleViewMatrix() function, for example.

        Parameters
        ----------
        view_matrix: [4x4]=[16] It is flatten tuple or list with 16 elements.

        Returns
        -------
        (   rgb_out,    : captured image
            depth_out,  : depth image converted to meters.
            mask_out,)  : Labeled mask image
        """

        ''' Reference of p.getCameraImage()
        The getCameraImage API will return a RGB image, a depth buffer and a segmentation mask
        buffer with body unique ids of visible objects for each pixel.
        returns:
            width:int 
            height:int
            rgbPixels:list of [char RED,char GREEN,char BLUE, char ALPHA] [0..width*height]
            depthPixels: list of float [0..width*height]
            segmentationMaskBuffer: list of int [0..width*height]
        '''
        w, h, rgb, depth, mask = p.getCameraImage(
            width=self._width,  # horizontal image resolution in pixels
            height=self._height,  # vertical image resolution in pixels
            viewMatrix=view_matrix,  # 4x4 view matrix, see PyBullet computeViewMatrix().
            projectionMatrix=self._projection_matrix,  # 4x4 projection matrix
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        if need_to_convert := (type(rgb) is tuple):
            rgb = np.asarray(rgb).reshape(self._height, self._width, 4)
            depth = np.asarray(depth).reshape(self._height, self._width)
            mask = np.asarray(mask).reshape(self._height, self._width)

        rgb_out = rgb[:, :, :3].astype(np.uint8)
        depth_out = self._z_far * self._z_near / (self._z_far - (self._z_far - self._z_near) * depth)
        mask_out = mask

        mask_out[mask_out == -1] = 0  # label empty space as 0

        return rgb_out, depth_out, mask_out
