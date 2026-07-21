from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "apple" / "new"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "apple_unprocessed_parquet"
INGEST_SCRIPT = PROJECT_ROOT / "src" / "data_ingestion" / "ingest_apple.py"


def session_id_from_filename(imubin_path: Path) -> str:
    """
    Expected filename examples:
      1_2026-05-14_5E150B30-957D-4B1C-888A-04659B2F1750.imubin
      1_2026-07-14_8F1B2AFC-E16B-41A0-9900-87C83F4E3C60_enzo.imubin

    The ingest script writes outputs using the run_id stored in the file header,
    which matches the UUID portion of these filenames.
    """
    stem = imubin_path.stem
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Could not derive session id from filename: {imubin_path.name}")

    return parts[2]


def already_ingested(imubin_path: Path) -> bool:
    session_id = session_id_from_filename(imubin_path)

    parquet_path = OUT_DIR / f"{session_id}.parquet"
    meta_path = OUT_DIR / f"{session_id}.meta.json"

    return parquet_path.exists() or meta_path.exists()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    imubin_files = sorted(RAW_DIR.glob("*.imubin"))

    if not imubin_files:
        print(f"No .imubin files found in {RAW_DIR}")
        return 0

    print(f"Found {len(imubin_files)} .imubin file(s).")

    ingested = 0
    skipped = 0
    failed = 0

    for imubin_path in imubin_files:
        try:
            if already_ingested(imubin_path):
                print(f"SKIP already ingested: {imubin_path.name}")
                skipped += 1
                continue

            print(f"INGEST {imubin_path.name}")

            result = subprocess.run(
                [
                    sys.executable,
                    str(INGEST_SCRIPT),
                    str(imubin_path),
                    str(OUT_DIR),
                ],
                cwd=PROJECT_ROOT,
                text=True,
            )

            if result.returncode == 0:
                ingested += 1
            else:
                failed += 1
                print(f"FAILED {imubin_path.name} with exit code {result.returncode}")

        except Exception as exc:
            failed += 1
            print(f"FAILED {imubin_path.name}: {exc}")

    print()
    print("Done.")
    print(f"  Ingested: {ingested}")
    print(f"  Skipped:  {skipped}")
    print(f"  Failed:   {failed}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
