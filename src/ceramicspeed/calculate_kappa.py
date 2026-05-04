"""
feature_selection.py
====================
Kappa (viscosity ratio) calculation for rolling bearing lubrication assessment.

Implements the method described in:
    Jakobsen et al., "Detecting insufficient lubrication in rolling bearings,
    using a low cost MEMS microphone to measure vibrations",
    Mechanical Systems and Signal Processing, Vol. 200, 2023.

Kappa is defined as the ratio of the actual kinematic viscosity of the lubricant
at operating temperature to the minimum required kinematic viscosity for adequate
film formation (ISO 281):

    kappa = nu / nu_1

where:
    nu   = kinematic viscosity of the lubricant at operating temperature [mm^2/s]
    nu_1 = minimum required kinematic viscosity at operating speed [mm^2/s]
"""

from __future__ import annotations

import numpy as np

def minimum_required_viscosity(rpm: float, d_pw: float) -> float:
    """Minimum required kinematic viscosity nu_1 per ISO 281.

    Parameters
    ----------
    rpm:
        Shaft rotational speed [rev/min].  Must be > 0.
    d_pw:
        Bearing mean pitch diameter [mm].

    Returns
    -------
    float
        nu_1 in [mm^2/s].
    """
    if rpm < 1000:
        return 45000 * rpm ** (-0.83) * d_pw ** (-0.5)
    else:
        return 4500 * rpm ** (-0.5) * d_pw ** (-0.5)


def lubricant_viscosity_at_temperature(
    temp_c: float,
    nu_40: float,
    nu_100: float
) -> float:
    """Kinematic viscosity at an arbitrary temperature via ASTM D341 (Walther equation).

    Uses the two reference viscosities (at 40 °C and 100 °C) to interpolate /
    extrapolate the viscosity at any operating temperature.

    Parameters
    ----------
    temp_c:
        Operating temperature [°C].
    nu_40:
        Kinematic viscosity at 40 °C [mm^2/s].
    nu_100:
        Kinematic viscosity at 100 °C [mm^2/s].

    Returns
    -------
    float
        Kinematic viscosity at *temp_c* [mm^2/s].
    """
    # Convert to Kelvin
    T1 = 40.0 + 273.15
    T2 = 100.0 + 273.15
    T_op = temp_c + 273.15

    # ASTM D341 double-log linear relationship
    A1 = np.log10(np.log10(nu_40 + 0.7))
    A2 = np.log10(np.log10(nu_100 + 0.7))

    B = (A2 - A1) / (np.log10(T2) - np.log10(T1))
    A = A1 - B * np.log10(T1)

    nu = 10.0 ** (10.0 ** (A + B * np.log10(T_op))) - 0.7
    return float(nu)


def calculate_kappa(
    rpm: float,
    temp_c: float,
    d_pw: float,
    nu_40: float,
    nu_100: float
) -> float:
    """Compute the viscosity ratio kappa for a rolling bearing.

    Parameters
    ----------
    rpm:
        Shaft rotational speed [rev/min].
    temp_c:
        Lubricant / bearing operating temperature [°C].
    d_pw:
        Bearing mean pitch diameter [mm].
    nu_40:
        Lubricant kinematic viscosity at 40 °C [mm^2/s].
    nu_100:
        Lubricant kinematic viscosity at 100 °C [mm^2/s].

    Returns
    -------
    float
        kappa = nu(T) / nu_1(n, d_pw).
    """
    nu = lubricant_viscosity_at_temperature(temp_c, nu_40, nu_100)
    nu_1 = minimum_required_viscosity(rpm, d_pw)
    return nu / nu_1
