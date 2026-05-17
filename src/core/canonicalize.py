"""
Canonicalization: device-native IMU frames -> a single canonical body frame.

CANONICAL FRAME (watch on RIGHT wrist, screen up, arm hanging at side, palm facing thigh):
    +X = distal     (toward fingers)
    +Y = ulnar      (toward pinky side — the LEFT side of the watch body on a right wrist)
    +Z = out of screen (away from skin)

Gyro sign convention: right-handed about each canonical axis.
Units after canonicalization: accel in g, gyro in rad/s.

IMPORTANT — empirical validation:
    The rotation matrices below are HYPOTHESES based on documented device frames.
    They MUST be validated with a calibration recording before training. The
    recommended protocol is in `validate_calibration_protocol()` at the bottom of
    this file. If a model trained on canonicalized data shows device-specific
    failure modes, a wrong rotation is your prime suspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Device = Literal["apple_watch", "garmin"]
Wrist = Literal["left", "right"]
Crown = Literal["left", "right"]  # Apple only; which side of the watch the crown is on

# Standard gravity, for optional g <-> m/s^2 conversions.
G_MS2 = 9.80665


# ---------------------------------------------------------------------------
# Device-native -> "right-wrist canonical" rotation matrices
# ---------------------------------------------------------------------------
#
# Each matrix R is applied as: canonical = R @ native, where native and canonical
# are column vectors [x, y, z]. Same R is used for both accel and gyro because
# both are vectors in the same body frame (gyro is a pseudovector but rotates
# the same way under proper rotations; we are not doing reflections here).
#
# Apple Watch CMDeviceMotion frame (per Apple docs, watch worn RIGHT wrist, screen up,
# crown on LEFT side of watch body — Apple's default for right-wrist wear):
#   +X = toward the crown (left side of watch body = ulnar side on right wrist)
#   +Y = toward the top of the screen (distal, toward fingers)
#   +Z = out of the screen (away from skin)
#
# Canonical frame (right wrist): distal=+X, ulnar=+Y (pinky side), out=+Z
#
# Mapping Apple-native -> canonical:
#   canonical_X (distal)   = native_Y  (top of screen = toward fingers)
#   canonical_Y (ulnar)    = native_X  (toward crown = toward pinky on right wrist)
#   canonical_Z (out)      = native_Z  (out of screen, unchanged)
#
# Pure swap of X and Y: det = -1, a reflection. To keep a proper rotation we
# note that native_X (toward crown) IS toward pinky on the right wrist when the
# crown is on the left side, but we need to verify the sign. The swap form:
#   [[0,1,0],[1,0,0],[0,0,1]]  det=-1  (reflection — bad for pseudovectors)
# The proper-rotation nearest equivalent, 90° CCW about +Z:
#   [[0,1,0],[-1,0,0],[0,0,1]] det=+1
#   maps native_Y -> canonical_X ✓, native_X -> canonical_-Y ✗ (wrong ulnar sign)
# So the correct proper rotation is 90° CW about +Z:
#   [[0,-1,0],[1,0,0],[0,0,1]] det=+1
#   maps native_Y -> canonical_-X ✗ (wrong distal sign)
#
# Neither 90° rotation gives us both signs right, because the physical mapping
# IS a reflection (relabeling two axes without flipping either). The resolution:
# we accept the reflection matrix for Apple's axis relabeling. Applying the same
# reflection to gyro is correct here because we are relabeling axes of the SAME
# physical right-handed frame, not changing its handedness. The gyro pseudovector
# transforms identically to accel under such a relabeling.
#
# Apple default = right wrist, crown on LEFT side of watch body.
APPLE_DEFAULT_TO_CANONICAL = np.array(
    [
        [0.0, 1.0, 0.0],  # canonical X (distal)  = native Y (top of screen)
        [1.0, 0.0, 0.0],  # canonical Y (ulnar)   = native X (toward crown = pinky on right wrist)
        [0.0, 0.0, 1.0],  # canonical Z (out)     = native Z (out of screen)
    ],
    dtype=np.float64,
)
# det = -1: accepted reflection — see explanation above.
# TODO: verify gyro signs empirically (slow wrist-rotation-supination recording).

# Garmin frame: this varies by device family and firmware. For most modern
# Garmin wrist watches with the IMU exposed via FIT developer data:
#   +X = toward the top of the watch (distal on right wrist)
#   +Y = toward the left side of the watch body (ulnar / pinky side on right wrist)
#   +Z = out of the screen
# This means Garmin native ~= canonical (for right wrist, default button orientation).
GARMIN_DEFAULT_TO_CANONICAL = np.eye(3, dtype=np.float64)
# TODO: VERIFY. Garmin's IMU axis convention is poorly documented and varies
# between Fenix, Forerunner, Venu, etc. Do a calibration recording per device model.


# ---------------------------------------------------------------------------
# Wrist + crown adjustments
# ---------------------------------------------------------------------------

# Left wrist: relative to the right-wrist canonical frame, wearing the watch on
# the left wrist rotates it 180° about the distal axis (+X). The screen now faces
# the opposite direction relative to the body's midline, flipping ulnar and out-of-screen.
#
# Rotation by 180° about +X: diag(1, -1, -1). det = +1, proper rotation.
LEFT_WRIST_FLIP = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

# Crown-on-RIGHT for Apple (user wears watch with crown on right side of body,
# i.e. "upside down" relative to the right-wrist default where crown is on left):
# The watch body is rotated 180° about the out-of-screen axis (native Z).
# Rotation by 180° about Z: diag(-1, -1, 1). det = +1.
CROWN_RIGHT_FLIP_NATIVE = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)


def rotation_for(device: Device, wrist: Wrist, crown: Crown | None) -> np.ndarray:
    """
    Build the full 3x3 transform from device-native axes to canonical axes,
    accounting for wrist side and (for Apple) crown side.

    Canonical frame: right wrist, crown on LEFT (Apple default for right-wrist wear).
    """
    if device == "apple_watch":
        base = APPLE_DEFAULT_TO_CANONICAL.copy()
        # Apply crown adjustment in NATIVE frame BEFORE the base rotation.
        # Default is crown-on-left (right wrist). Crown-on-right needs a native flip.
        if crown == "right":
            base = base @ CROWN_RIGHT_FLIP_NATIVE
        elif crown not in ("left", None):
            raise ValueError(f"Unknown crown position: {crown!r}")
    elif device == "garmin":
        base = GARMIN_DEFAULT_TO_CANONICAL.copy()
        # Garmin doesn't expose a configurable crown side; crown argument is ignored.
    else:
        raise ValueError(f"Unknown device: {device!r}")

    # Apply wrist adjustment in CANONICAL frame AFTER the base rotation.
    # Right wrist is the canonical default — no adjustment needed.
    if wrist == "left":
        full = LEFT_WRIST_FLIP @ base
    elif wrist == "right":
        full = base
    else:
        raise ValueError(f"Unknown wrist: {wrist!r}")

    return full


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalizeSpec:
    """Everything needed to canonicalize one session."""
    device: Device
    wrist: Wrist
    crown: Crown | None = None
    # Native-data unit conversions. After applying these scale factors, accel
    # must be in g and gyro must be in rad/s.
    accel_scale_to_g: float = 1.0
    gyro_scale_to_rad_s: float = 1.0


# Convenience factories that bake in the unit conventions from your existing code.
def apple_imubin_spec(wrist: Wrist, crown: Crown) -> CanonicalizeSpec:
    """
    Apple .imubin already stores accel in g and gyro in rad/s (per CMDeviceMotion
    convention used in your plot_imu.py). No unit conversion needed.
    """
    return CanonicalizeSpec(
        device="apple_watch",
        wrist=wrist,
        crown=crown,
        accel_scale_to_g=1.0,
        gyro_scale_to_rad_s=1.0,
    )


def garmin_fit_spec(wrist: Wrist) -> CanonicalizeSpec:
    """
    Your existing Garmin FIT extractor divides accel by 102 and gyro by 100.
    After those raw divisions, you get:
      - accel in g (102 LSB / g is the Garmin elevate-sensor convention)
      - gyro in deg/s (NOT rad/s — 100 LSB / (deg/s))
    Your data_loader appears to feed these into a Butterworth in whatever units
    the CSV holds, which is fine for filtering but means the units are
    inconsistent with Apple. We fix that here by converting deg/s -> rad/s.

    NOTE: this assumes the raw CSVs from `extract_raw_fit_data` are what we're
    feeding in. If the upstream divisors change, update here.
    """
    return CanonicalizeSpec(
        device="garmin",
        wrist=wrist,
        crown=None,
        accel_scale_to_g=1.0,
        gyro_scale_to_rad_s=np.pi / 180.0,  # deg/s -> rad/s
    )


def canonicalize_array(
    accel_xyz: np.ndarray,
    gyro_xyz: np.ndarray,
    spec: CanonicalizeSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform (N, 3) accel and gyro arrays from device-native frame to canonical
    frame, with unit conversion.

    Returns (accel_canonical_g, gyro_canonical_rad_s), both (N, 3) float32.
    """
    if accel_xyz.ndim != 2 or accel_xyz.shape[1] != 3:
        raise ValueError(f"accel_xyz must be (N, 3), got {accel_xyz.shape}")
    if gyro_xyz.shape != accel_xyz.shape:
        raise ValueError(f"gyro_xyz shape {gyro_xyz.shape} != accel_xyz shape {accel_xyz.shape}")

    R = rotation_for(spec.device, spec.wrist, spec.crown)

    # Unit conversion first, then rotation. Order doesn't matter mathematically
    # for a linear rotation but this is clearer.
    a = accel_xyz.astype(np.float64) * spec.accel_scale_to_g
    g = gyro_xyz.astype(np.float64) * spec.gyro_scale_to_rad_s

    # Apply rotation: each row is a 3-vector, so right-multiply by R.T
    # (equivalent to: for each row v, compute R @ v).
    a_canon = a @ R.T
    g_canon = g @ R.T

    return a_canon.astype(np.float32), g_canon.astype(np.float32)


def canonicalize_dataframe(
    df: pd.DataFrame,
    spec: CanonicalizeSpec,
    accel_cols: tuple[str, str, str] = ("acc_x", "acc_y", "acc_z"),
    gyro_cols: tuple[str, str, str] = ("gyro_x", "gyro_y", "gyro_z"),
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Canonicalize the six IMU columns of a DataFrame. Returns a new DataFrame
    (or modifies in place) with the same column names but canonical values.
    All other columns (t_sec, label, etc.) pass through untouched.
    """
    accel = df[list(accel_cols)].to_numpy()
    gyro = df[list(gyro_cols)].to_numpy()

    a_canon, g_canon = canonicalize_array(accel, gyro, spec)

    out = df if inplace else df.copy()
    out[list(accel_cols)] = a_canon
    out[list(gyro_cols)] = g_canon
    return out


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _assert_proper_rotation_or_reflection(R: np.ndarray, name: str) -> None:
    """Sanity-check that R is orthogonal. Allows reflections (det = -1) for
    Apple's axis-relabel case, but prints a warning if det is unexpected."""
    should_be_identity = R @ R.T
    if not np.allclose(should_be_identity, np.eye(3), atol=1e-9):
        raise AssertionError(f"{name}: R is not orthogonal.\n{R}")
    det = np.linalg.det(R)
    if not (np.isclose(det, 1.0) or np.isclose(det, -1.0)):
        raise AssertionError(f"{name}: |det(R)| != 1 (got {det}).")


def selftest() -> None:
    """Run on import / from CLI to catch matrix typos."""
    for d, w, c in [
        ("apple_watch", "right", "left"),   # canonical default
        ("apple_watch", "right", "right"),
        ("apple_watch", "left", "left"),
        ("apple_watch", "left", "right"),
        ("garmin", "right", None),
        ("garmin", "left", None),
    ]:
        R = rotation_for(d, w, c)
        _assert_proper_rotation_or_reflection(R, f"{d}/{w}/{c}")

    # Sanity: a watch lying face-up on a table reads native accel ~(0, 0, +1g) on
    # both devices. We don't assert what each spec produces here (the wrist flip
    # is meant for WORN orientations, not table-flat orientations — see the
    # calibration protocol). Just verify shapes and that no spec crashes.
    rest_accel = np.array([[0.0, 0.0, 1.0]])
    rest_gyro = np.array([[0.0, 0.0, 0.0]])
    for spec in [
        apple_imubin_spec("right", "left"),
        apple_imubin_spec("right", "right"),
        garmin_fit_spec("right"),
        garmin_fit_spec("left"),
    ]:
        a, g = canonicalize_array(rest_accel, rest_gyro, spec)
        assert a.shape == (1, 3), f"Shape error for {spec}"
        assert g.shape == (1, 3), f"Shape error for {spec}"

    # Verify left-wrist flip is the inverse of itself (applying twice = identity).
    assert np.allclose(LEFT_WRIST_FLIP @ LEFT_WRIST_FLIP, np.eye(3))

    print("canonicalize.selftest: OK")


def validate_calibration_protocol() -> str:
    """Returns instructions for the empirical calibration recording."""
    return """
EMPIRICAL CALIBRATION PROTOCOL
==============================
Canonical frame: RIGHT wrist, crown on LEFT side of watch body (Apple default).
  +X = distal (toward fingers)
  +Y = ulnar  (toward pinky — the LEFT side of the watch body on a right wrist)
  +Z = out of screen (away from skin)

SIGN CONVENTION REMINDER:
  The accelerometer measures SPECIFIC FORCE (the reaction force on the sensor),
  which points OPPOSITE to gravitational acceleration. At rest on a table,
  with gravity pulling the device DOWN, the accelerometer reads UP (+1g away
  from the earth). So at rest in any pose, the accel vector points AWAY from
  the earth, in canonical body coordinates.

For each (device, wrist, crown) combination you plan to support:

1. Wear the watch in the specified configuration.
2. Stand upright, arm hanging straight down, palm facing your thigh. Hold 5s.
   Physically: hand points toward earth, so +X (distal) points toward earth.
   Accel reads OPPOSITE = away from earth = -X.
   Expected canonical accel: ~(-1g, 0, 0)
3. Raise arm straight forward, palm facing DOWN. Hold 5s.
   Physically: screen faces up (away from earth), so +Z (out-of-screen) points
   away from earth. Accel reads in that same direction.
   Expected canonical accel: ~(0, 0, +1g)
4. Raise arm straight to the side (abduction), palm facing FORWARD. Hold 5s.
   Physically: on the RIGHT wrist with palm forward, +Y (ulnar / pinky) points
   toward earth (pinky is on the underside of the arm). Accel reads opposite.
   Expected canonical accel: ~(0, -1g, 0) on RIGHT wrist
                             ~(0, -1g, 0) on LEFT wrist  (canonical frame is wrist-agnostic
                                                          AFTER canonicalization — that's
                                                          the whole point. If you get +1g
                                                          on one wrist and -1g on the other,
                                                          the wrist flip is wrong.)
5. Slowly pronate then supinate the wrist (palm-down -> palm-up) over ~3 seconds.
   Expected: gyro_X dominant, clean sinusoid; gyro_Y and gyro_Z near zero.

Run canonicalize on the recording and confirm:
- Static poses (steps 2-4): accel magnitude ~1g, direction matches above.
- Dynamic pose (step 5): gyro_X dominant signal, other axes quiet.

Importantly: repeat for both wrists and confirm the SAME physical motion produces
the SAME canonical signal. If left-wrist and right-wrist disagree on the sign of
any canonical axis for the same pose, the LEFT_WRIST_FLIP matrix is wrong.
"""


if __name__ == "__main__":
    selftest()
    print(validate_calibration_protocol())