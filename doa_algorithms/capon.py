import numpy as np
from scipy import linalg


class Capon:
    def __init__(self, array, doas_list):
        super().__init__()
        self.array = array
        self.doas_list = doas_list

    def _calc_weights(self, R_inv, stv):
        w = R_inv @ stv
        w /= (stv.conj().T @ R_inv @ stv)
        return w

    def _calc_power(self, weights, signal):
        y = np.dot(weights.conj().T, signal.T)
        power = np.mean(np.abs(y)**2)
        return power

    def estimate(self, input_signal: np.ndarray):
        power = np.empty_like(self.doas_list, dtype=np.float32)
        R = input_signal.conj().T @ input_signal / input_signal.shape[0]
        try:
            R_inv = linalg.inv(R)
        except linalg.LinAlgError:
            raise ValueError("Failed to invert R.")

        for i, doa in enumerate(self.doas_list):
            stv = self.steering_vector(doa)[:, np.newaxis]

            w = self._calc_weights(R_inv, stv)
            power[i] = self._calc_power(w, input_signal)

        return power
