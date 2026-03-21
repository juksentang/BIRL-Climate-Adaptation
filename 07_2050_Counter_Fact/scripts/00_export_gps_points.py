"""
Export unique GPS points from birl_sample.parquet to CSV for GEE upload.

Output: data/gps_points_for_gee.csv
  Columns: point_idx, country, latitude, longitude
  Rows: ~4,436 unique GPS locations
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import BIRL_SAMPLE_PATH, DATA_DIR

import pandas as pd

def main():
    df = pd.read_parquet(BIRL_SAMPLE_PATH)
    print(f"Loaded birl_sample: {df.shape[0]:,} obs")

    # Get unique GPS points with country
    gps = (
        df[["gps_lat_final", "gps_lon_final", "country"]]
        .drop_duplicates(subset=["gps_lat_final", "gps_lon_final"])
        .reset_index(drop=True)
    )
    gps.index.name = "point_idx"
    gps = gps.reset_index()
    gps = gps.rename(columns={
        "gps_lat_final": "latitude",
        "gps_lon_final": "longitude",
    })

    out_path = DATA_DIR / "gps_points_for_gee.csv"
    gps.to_csv(out_path, index=False)
    print(f"Exported {len(gps):,} unique GPS points to {out_path}")
    print(f"By country: {gps['country'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
