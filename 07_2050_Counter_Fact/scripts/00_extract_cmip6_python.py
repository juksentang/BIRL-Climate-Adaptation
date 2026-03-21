"""
Extract CMIP6 2050 climate projections via GEE Python API.

Strategy: GPS-point-level extraction using reduceRegions in batches.
  - 4,436 unique GPS points × 5 GCMs × 2 SSPs × 2 periods
  - Growing-season statistics per country
  - Output: 20 CSVs in data/cmip6_raw/

For each GCM × scenario × period, computes per-point:
  - rainfall_gs_sum: growing-season total precipitation (mm)
  - rainfall_gs_cv: inter-annual CV of growing-season rainfall
  - tmean_gs: growing-season mean temperature (°C)
  - tmax_gs: growing-season mean daily max temperature (°C)
"""

import sys
import time
import logging
from pathlib import Path

import ee
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import (
    DATA_DIR, CMIP6_RAW_DIR, GROWING_SEASONS, GCMS, SSPS, COUNTRIES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("cmip6")

ee.Initialize()

# ── Load GPS points ──
GPS_PATH = DATA_DIR / "gps_points_for_gee.csv"
PERIODS = {
    "baseline": ("2005-01-01", "2014-12-31"),  # historical scenario
    "future":   ("2045-01-01", "2055-12-31"),   # SSP scenario (10 years centered on 2050)
}
# Note: NASA/GDDP-CMIP6 historical ends at 2014. For baseline we use 2005-2014.
# Future: 2045-2055 (centered on 2050, 10 years for stable climatology).

BATCH_SIZE = 300  # points per reduceRegions call


def build_points_fc(gps_df: pd.DataFrame) -> dict:
    """Build per-country GEE FeatureCollections from GPS DataFrame."""
    country_fcs = {}
    for country in COUNTRIES:
        sub = gps_df[gps_df["country"] == country]
        features = []
        for _, row in sub.iterrows():
            pt = ee.Geometry.Point([row["longitude"], row["latitude"]])
            feat = ee.Feature(pt, {
                "point_idx": int(row["point_idx"]),
                "country": row["country"],
            })
            features.append(feat)
        country_fcs[country] = ee.FeatureCollection(features)
        log.info(f"  {country}: {len(sub)} points")
    return country_fcs


def get_gs_months(country: str) -> tuple:
    """Return (month_list, cross_year) for a country's growing season."""
    gs = GROWING_SEASONS[country]
    return gs["months"], gs["cross_year"]


def compute_annual_gs_image(cmip_col, year, months, cross_year):
    """Compute growing-season rain_sum, tmean, tmax for one year."""
    year = ee.Number(year)

    if cross_year:
        # e.g. Malawi: Nov(yr-1) to Apr(yr)
        first_month = months[0]  # e.g. 11
        last_month = months[-1]  # e.g. 4
        part1 = (cmip_col
                 .filter(ee.Filter.calendarRange(year.subtract(1), year.subtract(1), 'year'))
                 .filter(ee.Filter.calendarRange(first_month, 12, 'month')))
        part2 = (cmip_col
                 .filter(ee.Filter.calendarRange(year, year, 'year'))
                 .filter(ee.Filter.calendarRange(1, last_month, 'month')))
        season = part1.merge(part2)
    else:
        first_month = months[0]
        last_month = months[-1]
        season = (cmip_col
                  .filter(ee.Filter.calendarRange(year, year, 'year'))
                  .filter(ee.Filter.calendarRange(first_month, last_month, 'month')))

    # Precipitation: pr is in kg/m²/s, multiply by 86400 to get mm/day, then sum
    rain_sum = season.select('pr').map(lambda img: img.multiply(86400)).sum()
    # Temperature: K → °C
    tmax_mean = season.select('tasmax').mean().subtract(273.15)
    tmin_mean = season.select('tasmin').mean().subtract(273.15)
    tmean = tmax_mean.add(tmin_mean).divide(2)

    return rain_sum, tmean, tmax_mean


def extract_for_config(gcm, scenario, period, country_fcs):
    """Extract climate stats for one GCM × scenario × period, all countries.

    Returns: list of dicts (one per point).
    """
    start_date, end_date = PERIODS[period]
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    # Build collection
    if period == "baseline":
        # Historical scenario
        cmip = (ee.ImageCollection('NASA/GDDP-CMIP6')
                .filter(ee.Filter.eq('model', gcm))
                .filter(ee.Filter.eq('scenario', 'historical'))
                .filterDate(start_date, end_date))
    else:
        cmip = (ee.ImageCollection('NASA/GDDP-CMIP6')
                .filter(ee.Filter.eq('model', gcm))
                .filter(ee.Filter.eq('scenario', scenario))
                .filterDate(start_date, end_date))

    all_results = []

    for country in COUNTRIES:
        months, cross_year = get_gs_months(country)
        fc = country_fcs[country]

        # Compute per-year growing season stats, then average across years
        years = list(range(start_year + (1 if cross_year else 0), end_year + 1))
        n_years = len(years)

        rain_images = []
        tmean_images = []
        tmax_images = []

        for yr in years:
            rain, tmean, tmax = compute_annual_gs_image(cmip, yr, months, cross_year)
            rain_images.append(rain)
            tmean_images.append(tmean)
            tmax_images.append(tmax)

        # Multi-year statistics
        rain_col = ee.ImageCollection(rain_images)
        rain_mean = rain_col.mean()
        rain_std = rain_col.reduce(ee.Reducer.stdDev())
        rain_cv = rain_std.divide(rain_mean.max(0.1))

        tmean_mean = ee.ImageCollection(tmean_images).mean()
        tmax_mean = ee.ImageCollection(tmax_images).mean()

        combined = (rain_mean.rename('rainfall_gs_sum')
                    .addBands(rain_cv.rename('rainfall_gs_cv'))
                    .addBands(tmean_mean.rename('tmean_gs'))
                    .addBands(tmax_mean.rename('tmax_gs')))

        # Extract to points in batches
        n_points = fc.size().getInfo()
        fc_list = fc.toList(n_points)

        for batch_start in range(0, n_points, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, n_points)
            batch_fc = ee.FeatureCollection(fc_list.slice(batch_start, batch_end))

            extracted = combined.reduceRegions(
                collection=batch_fc,
                reducer=ee.Reducer.mean(),
                scale=25000,
            )

            # Get results
            try:
                features = extracted.getInfo()['features']
                for feat in features:
                    props = feat['properties']
                    coords = feat['geometry']['coordinates']
                    all_results.append({
                        'point_idx': props.get('point_idx', -1),
                        'country': props.get('country', country),
                        'longitude': coords[0],
                        'latitude': coords[1],
                        'rainfall_gs_sum': props.get('rainfall_gs_sum'),
                        'rainfall_gs_cv': props.get('rainfall_gs_cv'),
                        'tmean_gs': props.get('tmean_gs'),
                        'tmax_gs': props.get('tmax_gs'),
                    })
            except Exception as e:
                log.error(f"  Batch {batch_start}-{batch_end} failed for {country}: {e}")

        log.info(f"    {country}: {n_points} points extracted")

    return all_results


def main():
    log.info("=" * 60)
    log.info("CMIP6 Extraction via GEE Python API")
    log.info("=" * 60)

    # Load GPS points
    gps_df = pd.read_csv(GPS_PATH)
    log.info(f"Loaded {len(gps_df)} GPS points")

    # Build FeatureCollections
    log.info("Building GEE FeatureCollections...")
    country_fcs = build_points_fc(gps_df)

    # Extract for each GCM × scenario × period
    total = len(GCMS) * len(SSPS) * len(PERIODS)
    count = 0

    for gcm in GCMS:
        for ssp in SSPS:
            for period in PERIODS:
                count += 1
                desc = f"cmip6_{gcm}_{ssp}_{period}"
                out_path = CMIP6_RAW_DIR / f"{desc}.csv"

                if out_path.exists():
                    log.info(f"[{count}/{total}] SKIP (exists): {desc}")
                    continue

                log.info(f"\n[{count}/{total}] Extracting: {desc}")
                t0 = time.time()

                try:
                    results = extract_for_config(gcm, ssp, period, country_fcs)
                    df = pd.DataFrame(results)
                    df.to_csv(out_path, index=False)
                    elapsed = time.time() - t0
                    log.info(f"  Saved: {out_path.name} ({len(df)} rows, {elapsed:.0f}s)")
                except Exception as e:
                    elapsed = time.time() - t0
                    log.error(f"  FAILED: {desc} ({elapsed:.0f}s): {e}")

    log.info("\n" + "=" * 60)
    log.info("CMIP6 extraction complete.")
    log.info(f"Files in {CMIP6_RAW_DIR}:")
    for f in sorted(CMIP6_RAW_DIR.glob("*.csv")):
        log.info(f"  {f.name} ({f.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
