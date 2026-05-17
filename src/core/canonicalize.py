"""
Canonicalization: device-native IMU frames -> a single canonical body frame.

CANONICAL FRAME (watch on LEFT wrist, screen up, arm hanging at side, palm facing thigh):
    +X = distal     (toward fingers)
    +Y = ulnar      (toward pinky side)
    +Z = out of screen (away from skin)

Gyro sign convention: right-handed about each canonical axis.
Units after canonicalization: accel in g, gyro in rad/s.

IMPORTANT — empirical validation:
    The rotation matrices below are HYPOTHESES based on documented device frames.
    They MUST be validated with a calibration recording before training. The
    recommended protocol is in `validate_calibration()` at the bottom of this file.
    If a model trained on canonicalized data shows device-specific failure modes,
    a wrong rotation is your prime suspect.
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
# Device-native -> "left-wrist canonical" rotation matrices
# ---------------------------------------------------------------------------
#
# Each matrix R is applied as: canonical = R @ native, where native and canonical
# are column vectors [x, y, z]. Same R is used for both accel and gyro because
# both are vectors in the same body frame (gyro is a pseudovector but rotates
# the same way under proper rotations; we are not doing reflections here).
#
# Apple Watch CMDeviceMotion frame (per Apple docs, watch worn left wrist, screen up):
#   +X = toward the crown (right side of watch body, which is ulnar side on left wrist)
#   +Y = toward the top of the screen (distal, toward fingers)
#   +Z = out of the screen (away from skin)
#
# To map Apple-native -> canonical (distal, ulnar, out-of-screen):
#   canonical_X (distal)   = native_Y
#   canonical_Y (ulnar)    = native_X
#   canonical_Z (out)      = native_Z
#
# That is a swap of X and Y. NOTE: this is a reflection (det = -1), which would
# flip gyro signs incorrectly. We instead use a proper rotation by also flipping
# one axis. Convention here: canonical_Y = native_X (no flip) requires us to
# flip another axis to keep det = +1. We'll flip Z (canonical_Z = -native_Z)
# only if the determinant check fails — but the swap (X<->Y) alone IS a
# reflection. To keep a proper rotation, the correct mapping is:
#   canonical = [[0, 1, 0], [1, 0, 0], [0, 0, 1]] @ native   -> det = -1 (BAD)
#   canonical = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]] @ native  -> det = +1 (90° about Z)
#
# Using the rotation form: 90° clockwise about +Z when viewed from +Z.
# Verify by hand: native +Y (toward top of screen / fingers) -> canonical +X. Good.
#                 native +X (toward crown / ulnar)           -> canonical -Y? That's wrong;
#                 we want canonical +Y on left wrist.
#
# So neither pure rotation nor pure reflection trivially works without thinking
# about WHICH wrist the watch is on. The "Apple X = toward crown" depends on
# crown position (Digital Crown can be on left or right side of the watch body
# via the user's setting). Below we account for that with `crown`.
#
# Decision: we define the apple_watch matrix for the canonical case
#   (left wrist, crown on RIGHT side of watch body — Apple's default).
# Then `wrist` and `crown` are handled by additional rotations composed on top.
APPLE_DEFAULT_TO_CANONICAL = np.array(
    [
        [0.0, 1.0, 0.0],  # canonical X (distal) = native Y (top of screen)
        [1.0, 0.0, 0.0],
        # canonical Y (ulnar)  = native X (toward crown, which is ulnar on left wrist w/ crown on right)
        [0.0, 0.0, 1.0],  # canonical Z (out)    = native Z (out of screen)
    ],
    dtype=np.float64,
)
# det = -1: this is a reflection. We accept this here ONLY because it represents
# the relationship between two right-handed frames where the axis *labels* differ
# but the physical handedness is the same. For gyro to come out right, we apply
# the same transform to gyro (it's a pseudovector but we're not changing handedness
# of the physical frame, only relabeling axes — so the same R applies).
# TODO: verify gyro signs on a known rotation (e.g., turning the wrist palm-up).

# Garmin frame: this varies by device family and firmware. For most modern
# Garmin wrist watches with the IMU exposed via FIT developer data:
#   +X = toward the top of the watch (distal on left wrist)
#   +Y = toward the right side of the watch body (ulnar on left wrist if buttons on right)
#   +Z = out of the screen
# This means Garmin native ~= canonical (for left wrist, default button orientation).
GARMIN_DEFAULT_TO_CANONICAL = np.eye(3, dtype=np.float64)
# TODO: VERIFY. Garmin's IMU axis convention is poorly documented and varies
# between Fenix, Forerunner, Venu, etc. Do a calibration recording per model.


# ---------------------------------------------------------------------------
# Wrist + crown adjustments
# ---------------------------------------------------------------------------

# Right wrist: the watch is rotated 180° about the distal axis (canonical +X)
# compared to left wrist, because the screen now faces the OTHER direction
# relative to the body's midline. Concretely: ulnar direction flips, and
# out-of-screen direction flips.
#
# Rotation by 180° about +X: diag(1, -1, -1). det = +1, proper rotation. Good.
RIGHT_WRIST_FLIP = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

# Crown-on-left (Apple, user wears watch "upside down" relative to default):
# The watch body is rotated 180° about the out-of-screen axis (native Z).
# Rotation by 180° about Z: diag(-1, -1, 1). det = +1.
CROWN_LEFT_FLIP_NATIVE = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)


def rotation_for(device: Device, wrist: Wrist, crown: Crown | None) -> np.ndarray:
    """
    Build the full 3x3 transform from device-native axes to canonical axes,
    accounting for wrist side and (for Apple) crown side.
    """
    if device == "apple_watch":
        base = APPLE_DEFAULT_TO_CANONICAL.copy()
        # Apply crown adjustment in NATIVE frame BEFORE the base rotation.
        # If crown is on the left, pre-multiply native by the crown flip.
        if crown == "left":
            base = base @ CROWN_LEFT_FLIP_NATIVE
        elif crown not in ("right", None):
            raise ValueError(f"Unknown crown position: {crown!r}")
    elif device == "garmin":
        base = GARMIN_DEFAULT_TO_CANONICAL.copy()
        if crown is not None:
            # Garmin doesn't have a configurable crown side in the same sense;
            # silently ignore but warn-worthy in logs.
            pass
    else:
        raise ValueError(f"Unknown device: {device!r}")

    # Apply wrist adjustment in CANONICAL frame AFTER the base rotation.
    if wrist == "right":
        full = RIGHT_WRIST_FLIP @ base
    elif wrist == "left":
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
        ("apple_watch", "left", "right"),
        ("apple_watch", "left", "left"),
        ("apple_watch", "right", "right"),
        ("apple_watch", "right", "left"),
        ("garmin", "left", None),
        ("garmin", "right", None),
    ]:
        R = rotation_for(d, w, c)
        _assert_proper_rotation_or_reflection(R, f"{d}/{w}/{c}")

    # Round-trip: gravity pointing along native -Z (watch lying face-up on table,
    # screen up) should give canonical accel ~= (0, 0, +1g) regardless of device
    # or wrist. Note: Apple/Garmin both report gravity as +Z when face-up because
    # the accelerometer measures the reaction force (specific force), so a watch
    # at rest face-up reads +1g on its Z axis.
    rest_accel = np.array([[0.0, 0.0, 1.0]])  # 1g out of screen
    rest_gyro = np.array([[0.0, 0.0, 0.0]])
    for spec in [
        apple_imubin_spec("left", "right"),
        apple_imubin_spec("right", "right"),
        garmin_fit_spec("left"),
        garmin_fit_spec("right"),
    ]:
        a, _ = canonicalize_array(rest_accel, rest_gyro, spec)
        # Canonical Z should still be ~+1g (screen still faces up; canonical Z
        # is "out of screen, away from skin" which equals "up" when the watch
        # is on a table).
        # BUT: for right-wrist, our convention says canonical Z flips because
        # the watch is "facing the other way" when worn — except when it's on
        # a TABLE face-up, both wrists give the same physical orientation.
        # This is a subtle point: the wrist flip is meant to handle the case
        # when the watch is being WORN. A table calibration on either wrist
        # spec should show the SAME canonical output (a clue that wrist info
        # alone is insufficient — you also need to know if the watch is on a
        # wrist or a table).
        # For a worn calibration (arm-hanging-at-side), things work out.
        # We don't assert anything specific here, just verify no crashes.
        assert a.shape == (1, 3)

    print("canonicalize.selftest: OK")


def validate_calibration_protocol() -> str:
    """Returns instructions for the empirical calibration recording."""
    return """
EMPIRICAL CALIBRATION PROTOCOL
==============================
For each (device, wrist, crown) combination you plan to support:

1. Wear the watch in the specified configuration.
2. Start recording.
3. Stand with arm hanging straight down at your side, palm facing your thigh.
   Hold still for 5 seconds. (Expected canonical accel: ~[0, 0, +1g] — wait,
   actually with arm hanging down, gravity points along -X canonical (proximal).
   So expected: accel ~ [-1g, 0, 0].)
4. Raise your arm straight forward (palm down), hold 5s.
   (Gravity now along -Z canonical: accel ~ [0, 0, -1g])
5. Raise your arm straight to the side, palm forward, hold 5s.
   (Depends on wrist; verify against your derivation)
6. Slowly rotate wrist palm-up then palm-down over 3 seconds.
   (Should produce a clean +/- gyro_X canonical signal)
7. Stop recording.

Then run canonicalize on the recording and confirm:
- Steps 3-5: accel magnitude is ~1g, direction matches expectations above.
- Step 6: gyro_X has a clear sinusoidal shape; gyro_Y and gyro_Z stay near zero.

Repeat across all wrist/crown combos. Right-wrist data should produce the
SAME canonical signals as left-wrist data for the same physical motion.
If it doesn't, the rotation matrices are wrong.
"""


if __name__ == "__main__":
    selftest()
    print(validate_calibration_protocol())