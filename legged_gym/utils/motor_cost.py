"""Motor torque<->mass cost model for motor-torque co-design (Path A).

Fitted to the ENCOS 电机质量-扭矩曲线 (peak / rated torque vs motor mass, read off the figure).
Used to couple the motor-torque DESIGN VARIABLE to a real mass cost: choosing a larger joint
torque limit (effort) requires a heavier motor, whose extra mass is added to that joint's link.
This is what makes motor co-design a genuine trade-off (no "free torque").

Reference anchors (current WF_TRON1A): leg joints effort 80 N·m peak -> ~690 g motor;
wheel joints effort 40 N·m -> ~400 g.
"""

import numpy as np

# Cubic fits: torque[N·m] = f(mass[g]), valid for mass in [384, 3100] g (the curve's sampled range).
# Anchored to ENCOS (南京因克斯智能科技 / encos.cn) datasheet points VERIFIED via web 2026-06-30:
#   EC-A4310-P2-36: 384 g, peak 36 N·m, rated 12 N·m, 87 rpm (model fit: 36 N·m -> 380 g, matches)
#   EC-A6408-P2-25: 604 g, peak 60 N·m, rated 20 N·m, 135 rpm
# ENCOS planetary joint-module series spans peak 36-400 N·m from ~384 g (high torque density, geared).
# Cross-checked vs comparable QDD/geared actuators (MyActuator RMD-X, CubeMars AK) — ENCOS sits on the
# high-torque-density (geared) end, as expected. Source datasheet is a scanned PDF (encos.cn).
_PEAK_COEF = [-1.34e-08, 6.39933e-05, 0.0517332803, 18.9899468433]   # 峰值扭矩 (peak torque)
_RATED_COEF = [-4.9e-09, 2.24916e-05, 0.01444226, 6.3800799692]      # 额定扭矩 (rated/continuous torque)
_MASS_MIN_G, _MASS_MAX_G = 384.0, 3100.0


def peak_torque_for_mass(mass_g):
    """Peak torque (N·m) deliverable by a motor of the given mass (g)."""
    return float(np.polyval(_PEAK_COEF, float(np.clip(mass_g, _MASS_MIN_G, _MASS_MAX_G))))


def rated_torque_for_mass(mass_g):
    """Rated/continuous torque (N·m) of a motor of the given mass (g)."""
    return float(np.polyval(_RATED_COEF, float(np.clip(mass_g, _MASS_MIN_G, _MASS_MAX_G))))


def mass_for_peak_torque_g(torque_nm):
    """Inverse: motor mass (g) needed to deliver a given PEAK torque (matches URDF 'effort' semantics).
    Numerically inverts the peak-torque fit over the valid mass range; clamps outside the curve."""
    grid = np.linspace(_MASS_MIN_G, _MASS_MAX_G, 6000)
    torques = np.polyval(_PEAK_COEF, grid)
    return float(grid[int(np.argmin(np.abs(torques - float(torque_nm))))])


def motor_mass_kg(peak_torque_nm):
    """Motor mass (kg) for a joint whose peak-torque limit is peak_torque_nm."""
    return mass_for_peak_torque_g(peak_torque_nm) / 1000.0


def motor_mass_delta_kg(base_effort_nm, scale):
    """Extra motor mass (kg, can be negative) when a joint's peak-torque limit is scaled from
    base_effort_nm to base_effort_nm * scale, per the ENCOS cost curve (relative to the nominal motor).

    Args:
        base_effort_nm: the joint's nominal peak-torque limit (URDF effort), e.g. 80 for legs, 40 for wheels.
        scale: design multiplier on the torque limit (k>1 stronger+heavier, k<1 lighter+weaker).
    Returns:
        delta mass in kg to add to that joint's link.
    """
    m1 = mass_for_peak_torque_g(float(base_effort_nm) * float(scale))
    m0 = mass_for_peak_torque_g(float(base_effort_nm))
    return (m1 - m0) / 1000.0


if __name__ == "__main__":
    print("torque[Nm] -> motor mass[g]")
    for t in [40, 60, 80, 100, 120, 150, 200]:
        print(f"  peak {t:4d} -> {mass_for_peak_torque_g(t):.0f} g")
    print("\nleg motor (80 N·m base) mass delta vs scale:")
    for k in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
        print(f"  k={k:.1f}  effort={80*k:.0f} N·m  delta_mass={motor_mass_delta_kg(80, k)*1000:+.0f} g")
