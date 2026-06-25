import numpy as np
import math

def _relativistic_electron_wavelength(v: float) -> float:
    return 1.23e-9 / np.sqrt(v + 1e-6 * v * v)

def _amplitude_contrast_phase_shift(q0: float) -> float:
    return math.atan(q0 / math.sqrt(1.0 - q0*q0))

class CtfContext:
    def __init__(
        self, 
        pixel_size_a: float, 
        spherical_aberration_mm: float, 
        voltage_kv: float,
        q0: float
    ):
        self.pixel_size_a = pixel_size_a
        self.spherical_aberration_mm = spherical_aberration_mm
        self.voltage_kv = voltage_kv
        self.q0 = q0

        self.spherical_aberration_a = spherical_aberration_mm * 1e7 
        self.wavelength_a = _relativistic_electron_wavelength(voltage_kv * 1e3) * 1e10
        self.amplitude_contrast_phase_shift = _amplitude_contrast_phase_shift(q0)
    