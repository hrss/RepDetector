"""
Canonicalization: device-native IMU frames -> a single canonical body frame.

CANONICAL FRAME (watch on RIGHT wrist, screen up, arm hanging at side, palm facing thigh):
    In the canonical frame, the normal force is:
    +X = proximal     (toward elbow, away from crown on default apple)
    +Y = ulnar      (toward pinky side — the LEFT side of the watch body on a right wrist)
    +Z = towards screen

Gyro sign convention: right-handed about each canonical axis (positive rotation
is clockwise when viewed looking in the +axis direction).

UNITS:
    By the time data reaches this module, it MUST already be in canonical units:
        accel: g
        gyro:  rad/s
    Unit conversion is the responsibility of the device-specific extractor
    upstream. This module performs rotations only.

IMPORTANT — empirical validation:
    The rotation matrices below encode the documented + empirically-observed
    axis conventions of each supported device. If a model trained on
    canonicalized data shows device-specific failure modes, the rotation for
    that device is the prime suspect. Re-run the calibration protocol in
    `validate_calibration_protocol()` to verify.
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

# Standard gravity, for callers that want to convert g <-> m/s^2 downstream.
G_MS2 = 9.80665


# ---------------------------------------------------------------------------
# Device-native -> canonical rotation matrices
# ---------------------------------------------------------------------------
#
# Each matrix R is applied as: canonical = R @ native, where native and canonical
# are column vectors [x, y, z]. The same R is used for both accel and gyro:
# both are vectors in the same body frame, and gyro (a pseudovector) transforms
# identically to accel under proper rotations (det = +1).

# Apple Watch CMDeviceMotion frame (right wrist, crown on LEFT side of watch body
# — Apple's default for right-wrist wear) IS the canonical frame, by definition.
APPLE_DEFAULT_TO_CANONICAL = np.eye(3, dtype=np.float64)

# Garmin frame (empirically observed): Garmin's X and Y axes point opposite to
# the canonical frame; Z agrees.
#   +X_garmin = proximal      (canonical -X)
#   +Y_garmin = radial        (canonical -Y)
#   +Z_garmin = out of screen (canonical +Z)
# This is a 180° rotation about +Z, det = +1, a proper rotation.
GARMIN_DEFAULT_TO_CANONICAL = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)


# ---------------------------------------------------------------------------
# Wrist + crown adjustments
# ---------------------------------------------------------------------------
#
# Both adjustments are 180° rotations and therefore self-inverse: applying
# the same flip twice returns the identity. Composition order matters for
# clarity (see rotation_for below): crown is applied in the NATIVE frame,
# wrist is applied in the CANONICAL frame.

# Left wrist vs. right wrist (canonical): wearing the watch on the opposite
# wrist mirrors motion along the distal axis. Concretely: +X (distal) flips,
# while +Y (ulnar) and +Z (out-of-screen) stay the same. This is a reflection
# (det = -1), not a proper rotation, because switching wrists really IS a
# mirror operation on the body — left and right arms are mirror images.
#
# A right-wrist motion's mirror on the left wrist is a reflected motion.
LEFT_WRIST_FLIP = np.diag([-1.0, 1.0, 1.0]).astype(np.float64)

LEFT_WRIST_GYRO_FLIP = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

# Apple Watch crown-on-RIGHT: user has rotated the watch 180° on their wrist
# compared to Apple's default orientation. This is a 180° rotation about the
# out-of-screen axis (native Z), which flips both native X and native Y.
# det = +1, proper rotation.
CROWN_RIGHT_FLIP_NATIVE = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)


def rotation_for(device: Device, wrist: Wrist, crown: Crown | None, flip_gyro: bool = False) -> np.ndarray:
    """
    Build the full 3x3 transform from device-native axes to canonical axes,
    accounting for wrist side and (for Apple) crown side.

    Canonical default: right wrist, Apple watch with crown on LEFT.

    Composition order:
        1. Crown flip (Apple only) — applied in NATIVE frame.
        2. Device base rotation — native -> canonical (right-wrist).
        3. Wrist flip — applied in CANONICAL frame.
    """
    if device == "apple_watch":
        base = APPLE_DEFAULT_TO_CANONICAL.copy()
        if crown == "right":
            base = base @ CROWN_RIGHT_FLIP_NATIVE
        elif crown not in ("left", None):
            raise ValueError(f"Unknown crown position: {crown!r}")
    elif device == "garmin":
        base = GARMIN_DEFAULT_TO_CANONICAL.copy()
        # Garmin doesn't expose a configurable crown side; argument is ignored.
    else:
        raise ValueError(f"Unknown device: {device!r}")

    if wrist == "left":
        if flip_gyro:
            full = LEFT_WRIST_GYRO_FLIP @ base
        else:
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
    """Everything needed to canonicalize one session.

    Input data MUST already be in canonical units (accel=g, gyro=rad/s). Unit
    conversion happens upstream in the device-specific extractor.
    """
    device: Device
    wrist: Wrist
    crown: Crown | None = None


def apple_imubin_spec(wrist: Wrist, crown: Crown) -> CanonicalizeSpec:
    """Apple .imubin: CMDeviceMotion already provides accel in g and gyro in rad/s."""
    return CanonicalizeSpec(device="apple_watch", wrist=wrist, crown=crown)


def garmin_fit_spec(wrist: Wrist) -> CanonicalizeSpec:
    """Garmin FIT: unit conversion (counts -> g, deg/s -> rad/s) is upstream."""
    return CanonicalizeSpec(device="garmin", wrist=wrist, crown=None)


def canonicalize_array(
    accel_xyz: np.ndarray,
    gyro_xyz: np.ndarray,
    spec: CanonicalizeSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate (N, 3) accel and gyro arrays from device-native frame to canonical
    frame. Input must already be in canonical units (accel=g, gyro=rad/s).

    Returns (accel_canonical, gyro_canonical), both (N, 3) float32.
    """
    if accel_xyz.ndim != 2 or accel_xyz.shape[1] != 3:
        raise ValueError(f"accel_xyz must be (N, 3), got {accel_xyz.shape}")
    if gyro_xyz.shape != accel_xyz.shape:
        raise ValueError(f"gyro_xyz shape {gyro_xyz.shape} != accel_xyz shape {accel_xyz.shape}")

    R = rotation_for(spec.device, spec.wrist, spec.crown, False)
    R = rotation_for(spec.device, spec.wrist, spec.crown, True)

    # Each row is a 3-vector; right-multiply by R.T to apply R to each row.
    a_canon = accel_xyz.astype(np.float64) @ R.T
    g_canon = gyro_xyz.astype(np.float64) @ R.T

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

def _assert_orthogonal(R: np.ndarray, name: str) -> None:
    """Sanity-check that R is orthogonal (|det| = 1). Allows reflections."""
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-9):
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
        _assert_orthogonal(R, f"{d}/{w}/{c}")

    # Apple default with right-wrist + crown-left should be exactly identity.
    R = rotation_for("apple_watch", "right", "left")
    assert np.allclose(R, np.eye(3)), f"Apple canonical default should be identity:\n{R}"

    # Shape sanity through the array path.
    rest_accel = np.array([[0.0, 0.0, 1.0]])
    rest_gyro = np.array([[0.0, 0.0, 0.0]])
    for spec in [
        apple_imubin_spec("right", "left"),
        apple_imubin_spec("right", "right"),
        apple_imubin_spec("left", "left"),
        apple_imubin_spec("left", "right"),
        garmin_fit_spec("right"),
        garmin_fit_spec("left"),
    ]:
        a, g = canonicalize_array(rest_accel, rest_gyro, spec)
        assert a.shape == (1, 3) and g.shape == (1, 3), f"Shape error for {spec}"

    # Self-inverse flips.
    assert np.allclose(LEFT_WRIST_FLIP @ LEFT_WRIST_FLIP, np.eye(3))
    assert np.allclose(CROWN_RIGHT_FLIP_NATIVE @ CROWN_RIGHT_FLIP_NATIVE, np.eye(3))

    # Cross-device consistency check: a "pure +X distal" canonical motion
    # should produce identical canonical output regardless of which device
    # recorded it. Apple native +X already equals canonical +X. Garmin native
    # X points the OPPOSITE way (proximal), so to record canonical +X on a
    # Garmin you'd see native -X.
    apple_native_plus_x = np.array([[1.0, 0.0, 0.0]])      # Apple native +X
    garmin_native_minus_x = np.array([[-1.0, 0.0, 0.0]])    # Garmin native -X
    a_apple, _ = canonicalize_array(
        apple_native_plus_x, np.zeros_like(apple_native_plus_x),
        apple_imubin_spec("right", "left"),
    )
    a_garmin, _ = canonicalize_array(
        garmin_native_minus_x, np.zeros_like(garmin_native_minus_x),
        garmin_fit_spec("right"),
    )
    assert np.allclose(a_apple, a_garmin), \
        f"Cross-device disagreement: apple={a_apple}, garmin={a_garmin}"

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
   Expected canonical accel: ~(0, -1g, 0) — SAME on both wrists after
   canonicalization. (That's the whole point: canonical is wrist-agnostic.
   If you get +1g on one wrist and -1g on the other, LEFT_WRIST_FLIP is wrong.)
5. Slowly pronate then supinate the wrist (palm-down -> palm-up) over ~3 seconds.
   Expected: gyro_X dominant, clean sinusoid; gyro_Y and gyro_Z near zero.

Run canonicalize on the recording and confirm:
- Static poses (steps 2-4): accel magnitude ~1g, direction matches above.
- Dynamic pose (step 5): gyro_X dominant signal, other axes quiet.

Importantly: repeat for BOTH wrists, BOTH devices, and confirm the SAME
physical motion produces the SAME canonical signal. Any disagreement points
at the matrix for that (device, wrist, crown) combination.
"""


if __name__ == "__main__":
    selftest()
    print(validate_calibration_protocol())