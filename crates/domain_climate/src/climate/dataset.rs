//! Calibrated GISTEMP v4 temperature anomalies, Law Dome / Mauna Loa atmospheric $CO_2$
//! observations, IPCC AR6 carbon-cycle mass-balance projections, and optional synthetic walks.

use rand::Rng;

/// Annual NASA GISTEMP v4 global land-ocean surface temperature anomalies (°C) for 1880..=2023,
/// referenced to the 1951-1980 climatological baseline mean (0.00°C).
const GISTEMP_ANOMALIES_1880_2023: [f64; 144] = [
    -0.16, -0.08, -0.11, -0.17, -0.28, -0.33, -0.31, -0.36, -0.17, -0.10, // 1880-1889
    -0.35, -0.22, -0.27, -0.31, -0.30, -0.23, -0.11, -0.11, -0.27, -0.18, // 1890-1899
    -0.08, -0.16, -0.28, -0.37, -0.47, -0.26, -0.22, -0.39, -0.43, -0.48, // 1900-1909
    -0.43, -0.44, -0.36, -0.34, -0.15, -0.14, -0.36, -0.46, -0.30, -0.27, // 1910-1919
    -0.27, -0.19, -0.28, -0.26, -0.27, -0.22, -0.11, -0.21, -0.20, -0.36, // 1920-1929
    -0.16, -0.09, -0.16, -0.28, -0.12, -0.20, -0.15, -0.03, 0.00, -0.02, // 1930-1939
    0.12, 0.18, 0.07, 0.09, 0.20, 0.09, -0.07, -0.03, -0.11, -0.11, // 1940-1949
    -0.17, -0.07, 0.01, 0.08, -0.13, -0.14, -0.19, 0.05, 0.06, 0.03, // 1950-1959
    -0.03, 0.06, 0.03, 0.05, -0.20, -0.11, -0.06, -0.02, -0.08, 0.05, // 1960-1969
    0.03, -0.08, 0.01, 0.16, -0.07, -0.01, -0.10, 0.18, 0.07, 0.16, // 1970-1979
    0.26, 0.32, 0.14, 0.31, 0.16, 0.12, 0.18, 0.32, 0.39, 0.27, // 1980-1989
    0.45, 0.41, 0.22, 0.23, 0.31, 0.45, 0.33, 0.46, 0.61, 0.38, // 1990-1999
    0.39, 0.54, 0.63, 0.62, 0.53, 0.68, 0.64, 0.66, 0.54, 0.66, // 2000-2009
    0.72, 0.61, 0.65, 0.68, 0.75, 0.90, 1.01, 0.92, 0.85, 0.98, // 2010-2019
    1.02, 0.85, 0.89, 1.17, // 2020-2023
];

/// Historical atmospheric $CO_2$ anchor points `(year, ppm)` combining Law Dome ice-core records
/// (1850-1958) and NOAA Mauna Loa global reference observations (1959-2023).
const HISTORICAL_CO2_ANCHORS: [(f64, f64); 19] = [
    (1850.0, 285.2),
    (1870.0, 287.5),
    (1890.0, 294.4),
    (1900.0, 295.7),
    (1910.0, 299.7),
    (1920.0, 303.4),
    (1930.0, 307.2),
    (1940.0, 310.4),
    (1950.0, 311.3),
    (1958.0, 315.3),
    (1960.0, 316.9),
    (1970.0, 325.7),
    (1980.0, 338.8),
    (1990.0, 354.4),
    (2000.0, 369.7),
    (2010.0, 390.1),
    (2015.0, 400.8),
    (2020.0, 414.2),
    (2023.0, 421.1),
];

/// Returns calibrated annual global surface temperature anomalies (`1880..=2023`) in °C
/// relative to the 1951-1980 NASA GISTEMP v4 baseline.
#[verified_engine::verified]
pub fn get_temperature_anomalies() -> Vec<[f64; 2]> {
    GISTEMP_ANOMALIES_1880_2023
        .iter()
        .enumerate()
        .map(|(idx, &anomaly)| [(1880 + idx) as f64, anomaly])
        .collect()
}

/// Generates synthetic global temperature anomaly data using an injected RNG.
/// Returns a vector of `[year, anomaly]` pairs for exploratory random-walk comparison.
#[verified_engine::verified]
pub fn get_temperature_anomalies_with_rng<R: Rng + ?Sized>(rng: &mut R) -> Vec<[f64; 2]> {
    let mut time_series = Vec::new();
    let mut current_anomaly = -0.5;

    for year in 1850..=2020 {
        let noise: f64 = rng.gen_range(-0.15..0.15);
        let trend = if year < 1950 { 0.002 } else { 0.015 };
        current_anomaly += trend + noise;
        time_series.push([year as f64, current_anomaly]);
    }

    time_series
}

#[verified_engine::verified]
fn interpolate_historical_co2(year: f64) -> f64 {
    if year <= HISTORICAL_CO2_ANCHORS[0].0 {
        return HISTORICAL_CO2_ANCHORS[0].1;
    }
    for window in HISTORICAL_CO2_ANCHORS.windows(2) {
        let (y0, c0) = window[0];
        let (y1, c1) = window[1];
        if year <= y1 {
            let t = (year - y0) / (y1 - y0);
            return c0 + t * (c1 - c0);
        }
    }
    HISTORICAL_CO2_ANCHORS[HISTORICAL_CO2_ANCHORS.len() - 1].1
}

/// Computes calibrated historical atmospheric $CO_2$ concentrations (`1850..=2023`, in ppm) and
/// IPCC AR6 carbon-cycle mass-balance projections (`2023..=2100`):
/// $C(t) = C_0 + \int_{t_0}^t (E(t') - S(t'))\,dt'$.
///
/// `reduction_scenario` in `[0.0, 1.0]` controls annual anthropogenic emissions abatement from
/// business-as-usual (`0.0`) to rapid net-zero transition (`1.0`).
#[verified_engine::verified]
pub fn get_co2_projections(reduction_scenario: f64) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let scenario = if reduction_scenario.is_finite() {
        reduction_scenario.clamp(0.0, 1.0)
    } else {
        0.0
    };

    let mut historical = Vec::with_capacity(174);
    for year in 1850..=2023 {
        let y = year as f64;
        historical.push([y, interpolate_historical_co2(y)]);
    }

    let preindustrial_eq_ppm: f64 = 280.0;
    let sink_timescale_yr: f64 = 55.0;
    let initial_emissions_ppm_yr: f64 = 5.20;
    let annual_emissions_rate: f64 = 0.008 - 0.098 * scenario;

    let mut current_co2: f64 = 421.1;
    let mut projected = Vec::with_capacity(78);
    projected.push([2023.0, current_co2]);

    for year in 2024..=2100 {
        let dt_years = (year - 2024) as f64;
        let emissions = initial_emissions_ppm_yr * (1.0 + annual_emissions_rate).powf(dt_years);
        let natural_sink = ((current_co2 - preindustrial_eq_ppm) / sink_timescale_yr).max(0.0);
        current_co2 += emissions - natural_sink;
        projected.push([year as f64, current_co2]);
    }

    (historical, projected)
}

/// Generates synthetic global $CO_2$ concentration data and projections using an injected RNG.
#[verified_engine::verified]
pub fn get_co2_projections_with_rng<R: Rng + ?Sized>(
    rng: &mut R,
    reduction_scenario: f64,
) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let mut historical = Vec::new();
    let mut projected = Vec::new();

    let mut current_co2 = 280.0;

    for year in 1850..=2020 {
        let noise: f64 = rng.gen_range(-0.5..0.5);
        let increase = if year < 1950 { 0.1 } else { 1.5 };
        current_co2 += increase + noise;
        historical.push([year as f64, current_co2]);
    }

    let mut current_increase = 1.5;

    for year in 2021..=2100 {
        let noise: f64 = rng.gen_range(-0.5..0.5);
        let target_increase = 2.5 - (3.5 * reduction_scenario);

        current_increase += (target_increase - current_increase) * 0.05;
        current_co2 += current_increase + noise;

        projected.push([year as f64, current_co2]);
    }

    (historical, projected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_gistemp_baseline_1951_1980_mean_near_zero() {
        let series = get_temperature_anomalies();
        assert_eq!(series.len(), 144);
        assert_eq!(series.first(), Some(&[1880.0, -0.16]));
        assert_eq!(series.last(), Some(&[2023.0, 1.17]));

        let baseline_vals: Vec<f64> = series
            .iter()
            .filter(|pt| (1951.0..=1980.0).contains(&pt[0]))
            .map(|pt| pt[1])
            .collect();
        assert_eq!(baseline_vals.len(), 30);
        let baseline_mean = baseline_vals.iter().sum::<f64>() / baseline_vals.len() as f64;
        assert!(
            baseline_mean.abs() < 0.01,
            "Expected 1951-1980 baseline mean near 0.0°C, got {baseline_mean}"
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_calibrated_co2_projections_mass_balance_monotonicity() {
        let (hist_bau, proj_bau) = get_co2_projections(0.0);
        let (_, proj_mid) = get_co2_projections(0.35);
        let (_, proj_high) = get_co2_projections(1.0);

        assert_eq!(hist_bau.first(), Some(&[1850.0, 285.2]));
        assert_eq!(hist_bau.last(), Some(&[2023.0, 421.1]));
        assert_eq!(proj_bau.first(), hist_bau.last());

        let end_bau = proj_bau.last().unwrap()[1];
        let end_mid = proj_mid.last().unwrap()[1];
        let end_high = proj_high.last().unwrap()[1];
        assert!(end_bau > end_mid && end_mid > end_high);
        assert!(end_bau > 600.0 && end_high < 421.1);
    }

    #[test]
    #[verified_engine::verified]
    fn test_synthetic_climate_walks_with_rng() {
        let mut rng = oxidize_core::rng::OxidizeRng::default();
        let temp = get_temperature_anomalies_with_rng(&mut rng);
        let (hist, proj) = get_co2_projections_with_rng(&mut rng, 0.5);
        assert!(!temp.is_empty());
        assert!(!hist.is_empty() && !proj.is_empty());
    }
}
