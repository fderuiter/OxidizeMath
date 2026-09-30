use rand::Rng;

/// Simulates a 1D Random Walk.
///
/// # Arguments
/// * `steps` - Number of steps N.
///
/// # Returns
/// * `f64` - Final position.
#[verified_engine::verified]
pub fn random_walk_1d(steps: usize) -> f64 {
    let mut rng = oxidize_core::rng::OxidizeRng::default();
    // allow(entropy_guard)
    random_walk_1d_with_rng(steps, &mut rng)
}

/// Simulates a 1D Random Walk using an injected RNG.
#[verified_engine::verified]
pub fn random_walk_1d_with_rng<R: Rng + ?Sized>(steps: usize, rng: &mut R) -> f64 {
    let mut position = 0.0;
    for _ in 0..steps {
        if rng.gen_bool(0.5) {
            position += 1.0;
        } else {
            position -= 1.0;
        }
    }
    position
}

/// Estimates the Diffusion Coefficient D.
///
/// Formula: D ~ <x^2> / (2t)
///
/// # Arguments
/// * `num_walks` - Number of walks M to average over.
/// * `time_steps` - Duration t of each walk.
///
/// # Returns
/// * `f64` - Estimated Diffusion Coefficient D.
#[verified_engine::verified]
pub fn estimate_diffusion_coefficient(num_walks: usize, time_steps: usize) -> f64 {
    let mut rng = oxidize_core::rng::OxidizeRng::default();
    estimate_diffusion_coefficient_with_rng(num_walks, time_steps, &mut rng)
}

/// Estimates the Diffusion Coefficient D using an injected RNG.
#[verified_engine::verified]
pub fn estimate_diffusion_coefficient_with_rng<R: Rng + ?Sized>(
    num_walks: usize,
    time_steps: usize,
    rng: &mut R,
) -> f64 {
    let mut sum_sq_displacement = 0.0;

    for _ in 0..num_walks {
        // allow(entropy_guard)
        let final_pos = random_walk_1d_with_rng(time_steps, rng);
        sum_sq_displacement += final_pos * final_pos;
    }

    let msd = sum_sq_displacement / num_walks as f64;
    // Time t corresponds to number of steps if we assume dt=1.
    msd / (2.0 * time_steps as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_diffusion() {
        // For Random Walk D ~ 0.5 (since dx=1, dt=1).
        let mut rng = oxidize_core::rng::OxidizeRng::default();
        let d = estimate_diffusion_coefficient_with_rng(1000, 100, &mut rng);
        assert!(
            (d - 0.5).abs() < 0.1,
            "Diffusion coefficient should be approx 0.5. Got {}",
            d
        );
    }
}
