/**
 * Step 07: Extract CMIP6 2050 Climate Projections from NASA/GDDP-CMIP6.
 *
 * Run in Google Earth Engine Code Editor:
 *   1. Upload gps_points_for_gee.csv as an Asset (Table)
 *   2. Set POINTS_ASSET to your asset path
 *   3. Run this script — it submits 20 batch export tasks to Drive
 *   4. Download CSVs from Drive folder "CMIP6_2050_extraction"
 *   5. Place CSVs in data/cmip6_raw/
 *
 * Output: 20 CSVs (5 GCMs × 2 SSPs × 2 periods)
 *   Each CSV: point_idx, country, latitude, longitude,
 *             rainfall_gs_sum, rainfall_gs_cv, tmean_gs, tmax_gs
 */

// ══════════════════════════════════════════════
// CONFIGURATION — Edit this section
// ══════════════════════════════════════════════

// Path to your uploaded GPS points asset
var POINTS_ASSET = 'users/YOUR_USERNAME/gps_points_for_gee';

var GCMS = ['ACCESS-CM2', 'MIROC6', 'MRI-ESM2-0', 'INM-CM5-0', 'IPSL-CM6A-LR'];
var SCENARIOS = ['ssp245', 'ssp585'];
var PERIODS = {
  'baseline': {start: '2005-01-01', end: '2023-12-31'},
  'future':   {start: '2040-01-01', end: '2060-12-31'}
};

// Growing season months per country
var GROWING_SEASONS = {
  'Ethiopia':  {months: [6,7,8,9],         crossYear: false},
  'Malawi':    {months: [11,12,1,2,3,4],   crossYear: true},
  'Mali':      {months: [6,7,8,9,10],      crossYear: false},
  'Nigeria':   {months: [4,5,6,7,8,9,10],  crossYear: false},
  'Tanzania':  {months: [11,12,1,2,3,4,5], crossYear: true},
  'Uganda':    {months: [3,4,5,6],         crossYear: false}
};

var BATCH_SIZE = 500;
var DRIVE_FOLDER = 'CMIP6_2050_extraction';

// ══════════════════════════════════════════════
// MAIN LOGIC
// ══════════════════════════════════════════════

var points = ee.FeatureCollection(POINTS_ASSET);
print('Total points:', points.size());

// Get unique countries from the points
var countries = points.aggregate_array('country').distinct().sort();
print('Countries:', countries);

/**
 * For one GCM × scenario × period, compute per-point growing-season statistics.
 * Returns a FeatureCollection with climate stats appended.
 */
function extractForConfig(gcm, scenario, period) {
  var startDate = PERIODS[period].start;
  var endDate = PERIODS[period].end;
  var startYear = parseInt(startDate.substring(0, 4));
  var endYear = parseInt(endDate.substring(0, 4));

  // Filter CMIP6 collection
  var cmip = ee.ImageCollection('NASA/GDDP-CMIP6')
    .filter(ee.Filter.eq('model', gcm))
    .filter(ee.Filter.eq('scenario', period === 'baseline' ? 'historical' : scenario))
    .filterDate(startDate, endDate);

  // For baseline period, historical scenario may only go to ~2014,
  // then switch to the SSP scenario for 2015-2023
  if (period === 'baseline') {
    var cmipHist = ee.ImageCollection('NASA/GDDP-CMIP6')
      .filter(ee.Filter.eq('model', gcm))
      .filter(ee.Filter.eq('scenario', 'historical'))
      .filterDate(startDate, '2014-12-31');
    var cmipSSP = ee.ImageCollection('NASA/GDDP-CMIP6')
      .filter(ee.Filter.eq('model', gcm))
      .filter(ee.Filter.eq('scenario', scenario))
      .filterDate('2015-01-01', endDate);
    cmip = cmipHist.merge(cmipSSP);
  }

  // Process each country separately (different growing seasons)
  var allResults = ee.FeatureCollection([]);

  var countryList = ee.List(['Ethiopia', 'Malawi', 'Mali', 'Nigeria', 'Tanzania', 'Uganda']);

  countryList.getInfo().forEach(function(country) {
    var gs = GROWING_SEASONS[country];
    var countryPoints = points.filter(ee.Filter.eq('country', country));
    var nPoints = countryPoints.size();

    // Build annual growing-season rainfall and temperature images
    var years = ee.List.sequence(startYear, endYear);

    var annualStats = years.map(function(yr) {
      yr = ee.Number(yr);
      var monthFilter;

      if (gs.crossYear) {
        // Cross-year: e.g., Nov(yr-1) to Apr(yr)
        var prevYrMonths = cmip
          .filter(ee.Filter.calendarRange(yr.subtract(1), yr.subtract(1), 'year'))
          .filter(ee.Filter.calendarRange(gs.months[0], 12, 'month'));
        var currYrMonths = cmip
          .filter(ee.Filter.calendarRange(yr, yr, 'year'))
          .filter(ee.Filter.calendarRange(1, gs.months[gs.months.length - 1], 'month'));
        monthFilter = prevYrMonths.merge(currYrMonths);
      } else {
        monthFilter = cmip
          .filter(ee.Filter.calendarRange(yr, yr, 'year'))
          .filter(ee.Filter.calendarRange(gs.months[0], gs.months[gs.months.length - 1], 'month'));
      }

      // Precipitation: sum of daily pr over growing season (kg/m²/s → mm/day already in GDDP)
      var rainSum = monthFilter.select('pr').sum();

      // Temperature: daily mean from tasmax and tasmin
      var tasmax = monthFilter.select('tasmax');
      var tasmin = monthFilter.select('tasmin');
      var tmeanImg = tasmax.mean().add(tasmin.mean()).divide(2).subtract(273.15);
      var tmaxImg = tasmax.mean().subtract(273.15);

      return ee.Feature(null, {
        'year': yr,
        'rain': rainSum,
        'tmean': tmeanImg,
        'tmax': tmaxImg
      });
    });

    // Compute multi-year statistics
    var rainImages = ee.ImageCollection(annualStats.map(function(f) {
      return ee.Feature(f).get('rain');
    }));
    var rainMean = rainImages.mean();
    var rainStd = rainImages.reduce(ee.Reducer.stdDev());
    var rainCV = rainStd.divide(rainMean.max(0.1));

    var tmeanImages = ee.ImageCollection(annualStats.map(function(f) {
      return ee.Feature(f).get('tmean');
    }));
    var tmeanMean = tmeanImages.mean();

    var tmaxImages = ee.ImageCollection(annualStats.map(function(f) {
      return ee.Feature(f).get('tmax');
    }));
    var tmaxMean = tmaxImages.mean();

    // Stack into single image
    var combined = rainMean.rename('rainfall_gs_sum')
      .addBands(rainCV.rename('rainfall_gs_cv'))
      .addBands(tmeanMean.rename('tmean_gs'))
      .addBands(tmaxMean.rename('tmax_gs'));

    // Extract to points in batches
    var countryResult = combined.reduceRegions({
      collection: countryPoints,
      reducer: ee.Reducer.mean(),
      scale: 25000  // CMIP6 ~25km resolution
    });

    allResults = allResults.merge(countryResult);
  });

  return allResults;
}

// ══════════════════════════════════════════════
// SUBMIT BATCH EXPORTS
// ══════════════════════════════════════════════

GCMS.forEach(function(gcm) {
  SCENARIOS.forEach(function(scenario) {
    ['baseline', 'future'].forEach(function(period) {
      var desc = 'cmip6_' + gcm + '_' + scenario + '_' + period;
      print('Submitting: ' + desc);

      var results = extractForConfig(gcm, scenario, period);

      Export.table.toDrive({
        collection: results,
        description: desc,
        folder: DRIVE_FOLDER,
        fileNamePrefix: desc,
        fileFormat: 'CSV',
        selectors: ['point_idx', 'country', 'latitude', 'longitude',
                     'rainfall_gs_sum', 'rainfall_gs_cv', 'tmean_gs', 'tmax_gs']
      });
    });
  });
});

print('All 20 export tasks submitted. Check Tasks tab for progress.');
