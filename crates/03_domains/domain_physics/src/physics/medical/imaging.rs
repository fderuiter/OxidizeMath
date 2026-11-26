/// Calculates Tracking Error for cine EPID.
///
/// $$ E_{EPID} = C_{target} - C_{field} $$
///
/// # Arguments
///
/// * `target_center` ($C_{target}$) - Position of the target center.
/// * `field_centroid` ($C_{field}$) - Position of the field centroid.
///
/// # Returns
///
/// * `f64` - The tracking error.
#[verified_engine::verified]
pub fn tracking_error(target_center: f64, field_centroid: f64) -> f64 {
    target_center - field_centroid
}
