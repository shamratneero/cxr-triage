"""
Run this BEFORE validate_vindr_cxr.py — reads just 5 DICOM files to check
that decompression works, so you find out about a missing library in 10
seconds instead of after the full 18,000-image run fails partway through.
"""

import pydicom
import os

VINDR_IMAGE_DIR = "E:/vinbigdata-chest-xray-abnormalities-detection/train"

def main():
    if not os.path.exists(VINDR_IMAGE_DIR):
        print(f"FOLDER NOT FOUND: {VINDR_IMAGE_DIR}")
        print("Check that you've placed the downloaded data at this path.")
        return

    files = [f for f in os.listdir(VINDR_IMAGE_DIR) if f.endswith('.dicom')][:5]
    if not files:
        print("No .dicom files found in that folder — check the path/extraction.")
        return

    print(f"Testing {len(files)} sample DICOM files...\n")
    success, failed = 0, 0
    for fname in files:
        path = os.path.join(VINDR_IMAGE_DIR, fname)
        try:
            ds = pydicom.dcmread(path)
            arr = ds.pixel_array  # this line triggers decompression
            print(f"  OK: {fname}  shape={arr.shape}  "
                  f"transfer_syntax={ds.file_meta.TransferSyntaxUID}")
            success += 1
        except Exception as e:
            print(f"  FAILED: {fname}  error={e}")
            failed += 1

    print(f"\n{success} succeeded, {failed} failed.")
    if failed > 0:
        print("\nIf you see a decompression error above, install:")
        print("  pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg")
        print("then re-run this check.")
    else:
        print("\nAll good — safe to run the full validate_vindr_cxr.py now.")


if __name__ == '__main__':
    main()
