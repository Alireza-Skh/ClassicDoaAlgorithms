import numpy as np
from scipy import linalg
from signal_model.antenna_response import FarField1DSource


class Esprit1D(FarField1DSource):
    """
    References:
        [1] R. Roy and T. Kailath, "ESPRIT-estimation of signal parameters via
        rotational invariance techniques," IEEE Transactions on Acoustics,
        Speech and Signal Processing, vol. 37, no. 7, pp. 984–995,
        Jul. 1989.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def estimate(self, angles: list, snr: int, displacement_vector=1, formulation='tls'):
        if displacement_vector < 1:
            raise ValueError(
                'displacement_vector must be a non-negative integer.')

        x = self.collect_plane_wave_response(angles, snr)
        y = self.collect_plane_wave_response_doublets(angles, snr)
        z = np.vstack([x, y])

        R = np.cov(z, rowvar=False)

        E = linalg.svd(R)[0]
        Es = E[:, :self.num_source]
        Esx = Es[:-displacement_vector, :]
        Esy = Es[displacement_vector:, :]

        if formulation == 'tls':
            Exy = np.hstack((Esx, Esy))
            Exy = Exy.conj().T @ Exy
            V = linalg.svd(Exy)[0]
            V12 = V[:self.num_source, self.num_source:]
            V22 = V[self.num_source:, self.num_source:]
            Phi = (-V12 / V22)
        elif formulation == 'ls':
            Esx_H = Esx.conj().T
            Phi = (Esx_H @ Esx) / (Esx_H @ Esy)
        else:
            raise ValueError("Formulation must be either 'ls' or 'tls'.")

        doa = linalg.eigvals(Phi)
        doa = np.arcsin(np.angle(doa) / (np.pi * displacement_vector))
        return -doa
