import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
from scipy.ndimage import gaussian_filter1d

# def find_spectrum_peaks(
#     theta: np.ndarray,
#     spectrum: np.ndarray,
#     angle_res: float,
#     num_sources: int
# ) -> np.ndarray:
#     spectrum /= np.max(spectrum)
#     bin_width = theta[1] - theta[0]
#     distance_bins = angle_res // bin_width
#     distance_bins = max(1, distance_bins)

#     angles = []
#     while len(angles) < num_sources:
#         index = np.where(spectrum == 1)[0].item()
#         angles.append(index)
#         spectrum[int(index-distance_bins/2):int(index+distance_bins/2)] = 0
#         spectrum /= np.max(spectrum)

#     return np.array(angles)


class SpectrumPeakFinder:
    def __init__(self, expected_peaks, filter_type='butterworth', filter_params=None):
        """
        Initialize the peak finder with iterative peak detection approach.

        Parameters:
        - expected_peaks: int, exact number of peaks to find
        - filter_type: str, type of filter ('butterworth', 'gaussian', 'savgol', 'none')
        - filter_params: dict, parameters for the chosen filter
        """
        self.expected_peaks = expected_peaks
        self.filter_type = filter_type

        default_params = {
            'butterworth': {'cutoff_freq': 0.1, 'order': 4},
            'gaussian': {'sigma': 2.0},
            'savgol': {'window_length': 11, 'polyorder': 3}
        }

        if filter_params is None:
            self.filter_params = default_params.get(filter_type, {})
        else:
            self.filter_params = filter_params

    def apply_lowpass_filter(self, spectrum):
        if self.filter_type == 'none':
            return spectrum.copy()

        elif self.filter_type == 'butterworth':
            cutoff = self.filter_params.get('cutoff_freq', 0.1)
            order = self.filter_params.get('order', 4)
            b, a = butter(order, cutoff, btype='low', analog=False)
            filtered_spectrum = filtfilt(b, a, spectrum)

        elif self.filter_type == 'gaussian':
            sigma = self.filter_params.get('sigma', 2.0)
            filtered_spectrum = gaussian_filter1d(spectrum, sigma=sigma)

        elif self.filter_type == 'savgol':
            window_length = self.filter_params.get('window_length', 11)
            polyorder = self.filter_params.get('polyorder', 3)

            if window_length % 2 == 0:
                window_length += 1
            window_length = max(window_length, polyorder + 1)
            window_length = min(window_length, len(spectrum))

            filtered_spectrum = savgol_filter(spectrum, window_length, polyorder)

        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

        return filtered_spectrum

    def find_single_peak_with_width(self, spectrum, min_prominence_ratio=0.1):
        if len(spectrum) == 0 or np.max(spectrum) <= 0:
            return 0, 0

        min_prominence = np.max(spectrum) * min_prominence_ratio
        peaks, properties = find_peaks(spectrum, prominence=min_prominence)

        if len(peaks) == 0:
            # If no peaks found with prominence, just take the maximum
            peak_idx = np.argmax(spectrum)
            peak_amp = spectrum[peak_idx]

            # Estimate width by finding where amplitude drops to half maximum
            half_max = peak_amp / 2
            left_idx = peak_idx
            right_idx = peak_idx

            # Search left
            while left_idx > 0 and spectrum[left_idx] > half_max:
                left_idx -= 1

            # Search right
            while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_max:
                right_idx += 1

            width = right_idx - left_idx
            return int(peak_idx), width

        peak_heights = spectrum[peaks]
        highest_peak_idx = peaks[np.argmax(peak_heights)]
        peak_amp = spectrum[highest_peak_idx]

        peak_pos = np.where(peaks == highest_peak_idx)[0][0]
        if 'widths' in properties:
            width = properties['widths'][peak_pos]
        else:
            half_max = peak_amp / 2
            left_idx = highest_peak_idx
            right_idx = highest_peak_idx

            while left_idx > 0 and spectrum[left_idx] > half_max:
                left_idx -= 1
            while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_max:
                right_idx += 1

            width = right_idx - left_idx

        left_bound = int(max(0, highest_peak_idx - width/2))
        right_bound = int(min(len(spectrum), highest_peak_idx + width/2 + 1))

        peak_region = spectrum[left_bound:right_bound]
        if len(peak_region) > 0 and np.sum(peak_region) > 0:
            indices_in_region = np.arange(left_bound, right_bound)
            weights = peak_region / np.sum(peak_region)
            mean_index = np.sum(indices_in_region * weights)
        else:
            mean_index = highest_peak_idx

        return int(mean_index), width

    def zero_out_peak_region(self, spectrum, peak_index, peak_width):
        spectrum_copy = spectrum.copy()

        left_bound = int(max(0, peak_index - peak_width))
        right_bound = int(min(len(spectrum), peak_index + peak_width + 1))
        spectrum_copy[left_bound:right_bound] = 0

        return spectrum_copy

    def find_peak_indices(self, spectrum, min_prominence_ratio=0.05, return_filtered_spectrum=False):
        filtered_spectrum = self.apply_lowpass_filter(spectrum)
        working_spectrum = filtered_spectrum.copy()

        peak_indices = []

        for i in range(self.expected_peaks):
            peak_idx, peak_width = self.find_single_peak_with_width(
                working_spectrum, min_prominence_ratio
            )

            peak_indices.append(peak_idx)

            working_spectrum = self.zero_out_peak_region(
                working_spectrum, peak_idx, peak_width
            )

        if peak_indices:
            sorted_order = np.argsort(peak_indices)
            peak_indices = [peak_indices[i] for i in sorted_order]

        if return_filtered_spectrum:
            return peak_indices, filtered_spectrum
        return peak_indices
