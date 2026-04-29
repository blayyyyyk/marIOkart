from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from desmume.emulator_mkds import MarioKart
from desmume.mkds.fx import FX32_SCALE_FACTOR
from scipy.interpolate import interp1d, splev, splprep

np.set_printoptions(precision=3, suppress=True)


def signed_angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.arctan2(np.linalg.det(np.stack([u, v])), u @ v)


class NKM:
    def __init__(self, emu: MarioKart):
        self.memory = emu.memory
        self.driver_status = self.memory.race_status.driverStatus

        _checkpoint_data = self.memory.checkpoint_data

        _map_data = self.memory.map_data

        self.checkpoints = np.ctypeslib.as_array(_checkpoint_data)

    @cached_property
    def cpoi_positions(self) -> np.ndarray:
        p1 = np.column_stack((self.checkpoints["x1"], self.checkpoints["z1"]))
        p2 = np.column_stack((self.checkpoints["x2"], self.checkpoints["z2"]))
        return np.stack((p1, p2), axis=1).astype(np.float32) * FX32_SCALE_FACTOR

    @cached_property
    def cpoi_angles(self) -> np.ndarray:
        mid = self.cpoi_positions.sum(axis=1) / 2
        mid_shift_l = np.roll(mid, shift=1, axis=0)
        mid_shift_r = np.roll(mid, shift=-1, axis=0)

        diff_l = mid_shift_l - mid
        diff_r = mid_shift_r - mid

        angles = np.zeros(diff_l.shape[0])
        for i in range(diff_l.shape[0]):  # this isn't good (performance-wise) but its cached so we chillin
            angles[i] = signed_angle(diff_l[i], diff_r[i])

        return angles

    def _get_spline_representation(self, index):
        assert index == 0 or index == 1, "index must be 0 or 1"
        pts = self.cpoi_positions[:, index]
        tck, _u = splprep(pts.T, s=0, per=1)

        # below, we are making t values more uniform along the surface of the spline
        u_fine = np.linspace(0, 1, 2000)
        coords_fine = splev(u_fine, tck)

        points_fine = np.column_stack(coords_fine)

        diffs = np.diff(points_fine, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        cum_distances = np.concatenate(([0], np.cumsum(distances)))

        cum_distances_norm = cum_distances / cum_distances[-1]

        distance_to_u = interp1d(cum_distances_norm, u_fine, kind="linear", bounds_error=False, fill_value=(0.0, 1.0))

        def arc_length_splev(t):
            t_wrapped = np.asarray(t) % 1.0
            u_corrected = distance_to_u(t_wrapped)
            return splev(u_corrected, tck)

        return arc_length_splev

    @cached_property
    def cpoi_spline_left(self):
        return self._get_spline_representation(0)

    @cached_property
    def cpoi_spline_right(self):
        return self._get_spline_representation(1)

    def get_logical_cpoi_index(self, index: int) -> int:
        n_entries = self.cpoi_positions.shape[0]
        return index % n_entries

    def get_cpoi_position(self, offset: int = 1, driver_id: int = 0):
        current_index = self.get_logical_cpoi_index(self.driver_status[driver_id].curCpoi + offset)
        return self.cpoi_positions[current_index]

    def get_cpoi_angle_discrete(self, offset: int = 1, driver_id: int = 0):
        current_index = self.get_logical_cpoi_index(self.driver_status[driver_id].curCpoi + offset)
        return self.cpoi_angles[current_index]

    def get_cpoi_position_interp(self, offset: float = 0.05, driver_id: int = 0):
        current_progress = float(self.driver_status[driver_id].lapProgress)
        current_progress = (current_progress + offset) % 1.0
        return np.stack([self.cpoi_spline_left(current_progress), self.cpoi_spline_right(current_progress)])

    def get_cpoi_angle_interp(self, front_offset: float = 0.05, mid_offset: float = 0.05, back_offset: float | None = None, driver_id: int = 0):
        if back_offset is None:
            back_offset = -front_offset

        cp_back = self.get_cpoi_position_interp(mid_offset + back_offset, driver_id)
        cp_mid = self.get_cpoi_position_interp(mid_offset, driver_id)
        cp_front = self.get_cpoi_position_interp(mid_offset + front_offset, driver_id)

        mid_back = cp_back.sum(axis=0) / 2
        mid_mid = cp_mid.sum(axis=0) / 2
        mid_front = cp_front.sum(axis=0) / 2

        diff_mid_back = mid_mid - mid_back
        diff_mid_front = mid_front - mid_mid

        angle = signed_angle(diff_mid_back, diff_mid_front)

        return angle

    def get_checkpoint_angle_error(self, front_offset: float = 0.05, mid_offset: float = 0.05, back_offset: float | None = None, driver_id: int = 0) -> float:
        if back_offset is None:
            back_offset = -front_offset

        checkpoint_angle = self.get_cpoi_angle_interp(front_offset, mid_offset, back_offset, driver_id)

        cp_back = self.get_cpoi_position_interp(mid_offset + back_offset, driver_id)
        cp_mid = self.get_cpoi_position_interp(mid_offset, driver_id)
        mid_back = cp_back.sum(axis=0) / 2
        mid_mid = cp_mid.sum(axis=0) / 2
        diff_mid_back = mid_mid - mid_back

        kart_angle = signed_angle(diff_mid_back, self.memory.driver_velocity[[0, 2]])

        return abs(kart_angle - checkpoint_angle)


def make_checkpoint_plot(pts: np.ndarray):
    # 1. Define your data (x and y coordinates)
    x, y = pts[:, 0], pts[:, 1]

    # 2. Create the plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="blue", marker="o", label="Data Points")

    # 3. Add labels and title for clarity
    plt.title("Simple 2D Point Plot")
    plt.xlabel("X-Axis Label")
    plt.ylabel("Y-Axis Label")

    # 4. Optional: Add a grid and legend
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # 5. Display the plot
    plt.show()
