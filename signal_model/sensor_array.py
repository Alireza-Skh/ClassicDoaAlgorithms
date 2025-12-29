import numpy as np
from numpy.random import default_rng
import torch


class ULA:
    C = 3e8

    def __init__(
        self,
        num_antenna: int,
        freq: int,
        num_samples: int,
        num_targets: int,
        angles_bound: list | tuple,
        element_spacing: float = 0.5,  # lambda / 2
        min_resolution: float = 1,
        baseband_mode: bool = True,
        coherent: bool = False,
        angle_type: str = "rad",
        array_imperfections: str = "none",
        rng=default_rng(seed=42),
    ):
        super().__init__()
        if angle_type.lower() not in ["rad", "deg"]:
            raise ValueError(
                f"Invalid angle type '{angle_type}'. Supported types are 'deg' and 'rad'.")
        if freq <= 0:
            raise ValueError("Frequency must be positive.")
        if num_antenna <= 0:
            raise ValueError("Number of antennas must be positive.")
        if element_spacing <= 0:
            raise ValueError("Element spacing must be positive.")
        if array_imperfections.lower() not in ("none", "gain", "phase", "pos", "all"):
            raise ValueError(
                f"Invalid array_imperfections type '{array_imperfections}'.")

        self._is_degrees = angle_type.lower() == "deg"
        self.freq = freq
        self.num_antenna = num_antenna
        self.num_samples = num_samples
        self.num_targets = num_targets
        self.baseband_mode = baseband_mode
        self.coherent = coherent
        self.element_spacing = element_spacing
        self.min_resolution = min_resolution
        self.angles_bound = angles_bound
        self.rng = rng
        self.angles_list = self._angles_list
        self.array_imperfections = array_imperfections.lower()

        self.d = self._tx_indices * self.element_spacing * self._lambda

        self.antenna_errors = np.ones(num_antenna, dtype=complex)
        if self.array_imperfections.lower() != "none":
            if "gain" in array_imperfections or "all" in array_imperfections:
                self.antenna_errors *= self.gain_error()
            if "phase" in array_imperfections or "all" in array_imperfections:
                self.antenna_errors *= self.phase_error()
            if "pos" in array_imperfections or "all" in array_imperfections:
                self.d += self.pos_error()

        if not self.baseband_mode:
            self.sampling_freq = 4 * self.freq
            self.sampling_interval = 1.0 / self.sampling_freq
            self.max_time = np.round(
                self.num_samples * self.sampling_interval, decimals=int(np.log10(self.freq))
            )

    @property
    def _lambda(self) -> float:
        return self.C / self.freq

    @property
    def _time(self) -> np.ndarray:
        return np.arange(self.num_samples) * self.sampling_interval

    @property
    def _tx_indices(self) -> np.ndarray:
        return np.arange(self.num_antenna)

    @property
    def _angles_list(self) -> np.ndarray:
        return np.linspace(*self.angles_bound, endpoint=False, dtype=np.float32)

    def gen_rand_angles(self) -> np.ndarray:
        shuffled = self.angles_list.copy()
        self.rng.shuffle(shuffled)

        keep = []
        for angle in shuffled:
            if not keep or all(abs(angle - k) >= self.min_resolution for k in keep):
                keep.append(angle)
            if len(keep) == self.num_targets:
                break

        if len(keep) < self.num_targets:
            raise ValueError(
                f"Could not find {self.num_targets} angles with minimum resolution {self.min_resolution}")

        return np.array(sorted(keep))

    def check_angle_type(self, angle) -> float:
        if self._is_degrees:
            angle = np.deg2rad(angle)
        return angle

    def gain_error(self) -> np.ndarray:
        error_db = (self.rng.random(self.num_antenna) - 0.5) * 6  # ±3dB
        error_linear = 10**(error_db / 20)
        error_linear[0] = 1.0
        return error_linear

    def phase_error(self) -> np.ndarray:
        error_deg = (self.rng.random(self.num_antenna) - 0.5) * 20  # ±10°
        error_rad = np.deg2rad(error_deg)
        error_rad[0] = 0.0
        return np.exp(1j * error_rad)

    def pos_error(self) -> np.ndarray:
        error = (self.rng.random(self.num_antenna) - 0.5) * \
            self.element_spacing * self._lambda  # ±0.5λ
        error[0] = 0.0
        return error

    def receive_steering_vector(self, angle: float) -> np.ndarray:
        angle = self.check_angle_type(angle)
        stv = np.exp(-2j * np.pi * self.d * np.sin(angle) / self._lambda)

        if self.array_imperfections.lower() != "none":
            stv *= self.antenna_errors

        return stv

    def receive_steering_vector_cuda(self, angle: float) -> torch.Tensor:
        device = angle.device

        if self._is_degrees:
            angle = torch.deg2rad(angle)

        d = torch.as_tensor(self.d).to(device)
        stv = torch.exp(-2j * torch.pi * d *
                        torch.sin(angle) / self._lambda).to(device)

        if self.array_imperfections.lower() != "none":
            stv *= torch.as_tensor(self.antenna_errors).to(device)

        return stv

    def steering_matrix(self, angles: np.ndarray) -> np.ndarray:
        angles = self.check_angle_type(angles)
        stm = np.exp(-2j * np.pi * self.d[:, None]
                     * np.sin(angles)[None, :] / self._lambda)

        if self.array_imperfections.lower() != "none":
            stm *= np.tile(self.antenna_errors[:, None], (1, self.num_targets))

        return stm

    def steering_matrix_derivative(self, angles: np.ndarray) -> np.ndarray:
        angles = self.check_angle_type(angles)
        st_v = np.exp(-2j * np.pi * self.d[:, None]
                      * np.sin(angles)[None, :] / self._lambda)
        return -2j * np.pi * self.d[:, None] * np.cos(angles)[None, :] * st_v / self._lambda

    def doublet_phase_delays_matrix(self, angles: np.ndarray) -> np.ndarray:
        angles = self.check_angle_type(angles)

        num_angles = angles.shape[0]
        phi = np.zeros((num_angles, num_angles), dtype=complex)
        for i, angle in enumerate(angles):
            phi[i, i] = np.exp(1j * np.sin(angle))

        return phi

    def random_noise(self, shape=None):
        if shape is None:
            shape = (self.num_antenna, self.num_samples)
        return (self.rng.standard_normal(shape) + 1j * self.rng.standard_normal(shape)) / np.sqrt(2)

    def p_signal(self, snr_db: float, shape=None):
        if shape is None:
            shape = (self.num_targets, self.num_samples)

        if self.baseband_mode:
            real_part = self.rng.standard_normal(shape)
            imaginary_part = self.rng.standard_normal(shape)
            sig = real_part + 1j * imaginary_part
        else:
            random_phases = self.rng.uniform(0, np.pi, self.num_targets)
            random_amplitudes = self.rng.uniform(0.1, 1.0, self.num_targets)
            sig = (
                np.cos(2 * np.pi * self.freq *
                       self._time + random_phases[:, np.newaxis])
                * random_amplitudes[:, np.newaxis]
            )

        if self.coherent:
            sig = np.tile(sig[0, :], (self.num_targets, 1))

        norms = np.linalg.norm(sig, keepdims=True)
        sig *= np.sqrt(10 ** (snr_db * 0.1)) / norms
        return sig

    def receive_waveform(self, target_angles: np.ndarray, snr_db: float) -> np.ndarray:
        stv = self.steering_matrix(target_angles)
        r_sig = np.dot(stv, self.p_signal(snr_db))
        noise = self.random_noise()
        return (r_sig + noise).T
