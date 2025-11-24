import numpy as np
import math
from scipy.spatial.transform import Rotation as R

# ======================= 공통 DH 함수 =======================
def get_dh_matrix(a, d, alpha, theta):
    a_r, t_r = map(math.radians, (alpha, theta))
    ca, sa, ct, st = np.cos(a_r), np.sin(a_r), np.cos(t_r), np.sin(t_r)
    return np.array([
        [ ct, -st*ca,  st*sa, a*ct],
        [ st,  ct*ca, -ct*sa, a*st],
        [  0,     sa,     ca,    d],
        [  0,      0,      0,    1]
    ])

def get_modified_dh_matrix(a, d, alpha, theta):
    a_r, t_r = map(math.radians, (alpha, theta))
    ca, sa, ct, st = np.cos(a_r), np.sin(a_r), np.cos(t_r), np.sin(t_r)
    return np.array([
        [ ct, -st,  0, a],
        [ st*ca, ct*ca, -sa, -d*sa],
        [ st*sa, ct*sa,  ca,  d*ca],
        [ 0, 0, 0, 1]
    ])

# ======================= 베이스 클래스 =======================
class RobotKinematics:
    def __init__(self, name):
        self.name = name
    
    def forward_kinematics(self, joint_angles, view=None):
        raise NotImplementedError

    def _truncate_angles(self, joint_angles):
        return joint_angles[:len(self.dh_params)]

# ======================= Meca500 =======================
class MecaInsertionKinematics(RobotKinematics):
    def __init__(self):
        super().__init__("MecaInsertion")
        self.dh_params = [
            {'alpha': -90, 'a': 0,     'd': 0.135, 'theta_offset': 0},
            {'alpha': 0,   'a': 0.135, 'd': 0,     'theta_offset': -90},
            {'alpha': -90, 'a': 0.038, 'd': 0,     'theta_offset': 0},
            {'alpha': 90,  'a': 0,     'd': 0.120, 'theta_offset': 0},
            {'alpha': -90, 'a': 0,     'd': 0,     'theta_offset': 0},
            {'alpha': 0,   'a': 0,     'd': 0.070, 'theta_offset': 0}
        ]
        # 베이스 보정
        rot_x_180 = R.from_euler('x', 180, degrees=True)
        rot_z_90 = R.from_euler('z', 90, degrees=True)
        self.base_correction = (rot_z_90 * rot_x_180).as_matrix()

    def forward_kinematics(self, joint_angles, view=None):
        joint_coords = [np.array([0,0,0])]
        T_cumulative = np.eye(4)
        T_cumulative[:3,:3] = self.base_correction
        base_point = np.array([[0],[0],[0],[1]])
        for i, params in enumerate(self.dh_params):
            theta = math.degrees(joint_angles[i]) + params['theta_offset']
            T_i = get_dh_matrix(params['a'], params['d'], params['alpha'], theta)
            T_cumulative = T_cumulative @ T_i
            joint_coords.append((T_cumulative @ base_point)[:3,0])
        return np.array(joint_coords, dtype=np.float32)

class Meca500Kinematics(RobotKinematics):
    def __init__(self):
        super().__init__("Meca500")
        self.dh_params = [
            {'alpha': -90, 'a': 0,     'd': 0.135, 'theta_offset': 0},
            {'alpha': 0,   'a': 0.135, 'd': 0,     'theta_offset': -90},
            {'alpha': -90, 'a': 0.038, 'd': 0,     'theta_offset': 0},
            {'alpha': 90,  'a': 0,     'd': 0.120, 'theta_offset': 0},
            {'alpha': -90, 'a': 0,     'd': 0,     'theta_offset': 0},
            {'alpha': 0,   'a': 0,     'd': 0.070, 'theta_offset': 0}
        ]
    def forward_kinematics(self, joint_angles, view=None):
        joint_coords_3d = [np.array([0, 0, 0])] # J0 (베이스)
        T_cumulative = np.eye(4)
        base_point = np.array([[0], [0], [0], [1]])
        for i in range(6):
            params = self.dh_params[i]
            theta = math.degrees(joint_angles[i]) + params['theta_offset']
            T_i = get_dh_matrix(params['a'], params['d'], params['alpha'], theta)
            T_cumulative = T_cumulative @ T_i
            joint_pos = T_cumulative @ base_point
            joint_coords_3d.append(joint_pos[:3, 0])
        return np.array(joint_coords_3d, dtype=np.float32)

# ======================= Franka Research 3 =======================
class Research3Kinematics(RobotKinematics):
    def __init__(self):
        super().__init__("research3")
        self.dh_params = [
            {'a': 0,      'd': 0.333, 'alpha':   0, 'theta_offset': 0}, 
            {'a': 0,      'd': 0,     'alpha': -90, 'theta_offset': 0},
            {'a': 0,      'd': 0.316, 'alpha':  90, 'theta_offset': 0},
            {'a': 0.0825, 'd': 0,     'alpha':  90, 'theta_offset': 0},
            {'a':-0.0825, 'd': 0.384, 'alpha': -90, 'theta_offset': 0},
            {'a': 0,      'd': 0,     'alpha':  90, 'theta_offset': 0},
            {'a': 0.088,  'd': 0,     'alpha':  90, 'theta_offset': 0}, 
            {'a': 0,      'd': 0.107, 'alpha':   0, 'theta_offset': 0}
        ]
        self.view_rotations = {
            'view1': R.from_euler('zyx',[90,180,0],degrees=True),
            'view2': R.from_euler('zyx',[90,180,0],degrees=True),
            'view3': R.from_euler('zyx',[90,180,0],degrees=True),
            'view4': R.from_euler('zyx',[90,180,0],degrees=True)
        }
        # 제외할 인덱스 (0부터 시작, base point 포함)
        self.exclude_indices = {1, 5}  

    def forward_kinematics(self, joint_angles, view="view1"):
        joint_coords = [np.array([0,0,0])]
        T_cumulative = np.eye(4)
        if view in self.view_rotations:
            T_cumulative[:3,:3] = self.view_rotations[view].as_matrix()
        base_point = np.array([[0],[0],[0],[1]])
        for i, angle_rad in enumerate(joint_angles):
            params = self.dh_params[i]
            theta_deg = math.degrees(angle_rad) + params['theta_offset']
            T_i = get_modified_dh_matrix(params['a'], params['d'], params['alpha'], theta_deg)
            T_cumulative = T_cumulative @ T_i
            joint_coords.append((T_cumulative @ base_point)[:3,0])
        # 제외할 인덱스 제거 후 반환
        filtered_coords = [pt for j, pt in enumerate(joint_coords) if j not in self.exclude_indices]
        return np.array(filtered_coords, dtype=np.float32)


# ======================= Panda franka =======================
class PandaKinematics(RobotKinematics):
    def __init__(self):
        super().__init__("panda")
        self.dh_params = [
            {'a': 0,      'd': 0.333, 'alpha':   0, 'theta_offset': 0}, 
            {'a': 0,      'd': 0,     'alpha': -90, 'theta_offset': 0},
            {'a': 0,      'd': 0.316, 'alpha':  90, 'theta_offset': 0},
            {'a': 0.0825, 'd': 0,     'alpha':  90, 'theta_offset': 0},
            {'a':-0.0825, 'd': 0.384, 'alpha': -90, 'theta_offset': 0},
            {'a': 0,      'd': 0,     'alpha':  90, 'theta_offset': 0},
            {'a': 0.088,  'd': 0,     'alpha':  90, 'theta_offset': 0}, 
            {'a': 0,      'd': 0.107, 'alpha':   0, 'theta_offset': 0}
        ]
        self.view_rotations = {
            'view1': R.from_euler('zyx',[90,180,0],degrees=True),
            'view2': R.from_euler('zyx',[90,180,0],degrees=True),
            'view3': R.from_euler('zyx',[90,180,0],degrees=True),
            'view4': R.from_euler('zyx',[90,180,0],degrees=True)
        }
        self.exclude_indices = {1, 5}

    def forward_kinematics(self, joint_angles, view="view1"):
        joint_coords = [np.array([0,0,0])]
        T_cumulative = np.eye(4)
        if view in self.view_rotations:
            T_cumulative[:3,:3] = self.view_rotations[view].as_matrix()
        base_point = np.array([[0],[0],[0],[1]])
        for i, angle_rad in enumerate(joint_angles):
            params = self.dh_params[i]
            theta_deg = math.degrees(angle_rad) + params['theta_offset']
            T_i = get_modified_dh_matrix(params['a'], params['d'], params['alpha'], theta_deg)
            T_cumulative = T_cumulative @ T_i
            joint_coords.append((T_cumulative @ base_point)[:3,0])
        filtered_coords = [pt for j, pt in enumerate(joint_coords) if j not in self.exclude_indices]
        return np.array(filtered_coords, dtype=np.float32)


# ======================= FR5 =======================
class Fr5Kinematics(RobotKinematics):
    def __init__(self):
        super().__init__("Fr5")
        self.dh_params = [
            {'alpha': 90,  'a': 0,      'd': 0.152, 'theta_offset': 0},
            {'alpha': 0,   'a': -0.425, 'd': 0,     'theta_offset': 0},
            {'alpha': 0,   'a': -0.395, 'd': 0,     'theta_offset': 0},
            {'alpha': 90,  'a': 0,      'd': 0.102, 'theta_offset': 0},
            {'alpha':-90,  'a': 0,      'd': 0.102, 'theta_offset': 0},
            {'alpha': 0,   'a': 0,      'd': 0.100, 'theta_offset': 0}
        ]
        self.view_rotations = {
            'top':   R.from_euler('zyx',[-85,0,180],degrees=True),
            'left':  R.from_euler('zyx',[180,0,90], degrees=True),
            'right': R.from_euler('zyx',[0,0,90],   degrees=True)
        }

    def forward_kinematics(self, joint_angles, view="top"):
        joint_coords = [np.array([0,0,0])]
        T_cumulative = np.eye(4)
        if view in self.view_rotations:
            T_cumulative[:3,:3] = self.view_rotations[view].as_matrix()
        base_point = np.array([[0],[0],[0],[1]])
        for i, params in enumerate(self.dh_params):
            theta = math.degrees(joint_angles[i]) + params['theta_offset']
            T_i = get_dh_matrix(params['a'], params['d'], params['alpha'], theta)
            T_cumulative = T_cumulative @ T_i
            joint_coords.append((T_cumulative @ base_point)[:3,0])
        return np.array(joint_coords, dtype=np.float32)

# ======================= 팩토리 함수 =======================
ROBOT_CLASSES = {
    "Meca500": Meca500Kinematics,
    "MecaInsertion": MecaInsertionKinematics,
    "research3": Research3Kinematics,
    "Fr5":     Fr5Kinematics,
    "panda":   PandaKinematics}

def get_robot_kinematics(robot_name):
    if robot_name not in ROBOT_CLASSES:
        raise ValueError(f"Unknown robot: {robot_name}")
    return ROBOT_CLASSES[robot_name]()
