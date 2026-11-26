use crate::ai::sds::diffusion::{NoiseSchedule, inject_noise};
use approx::assert_relative_eq;
use nalgebra::DMatrix;

#[test]
#[verified_engine::verified]
fn test_noise_schedule() {
    let steps = 100;
    let schedule = NoiseSchedule::new(steps);

    assert_eq!(schedule.betas.len(), steps);
    assert_eq!(schedule.alphas.len(), steps);
    assert_eq!(schedule.alpha_bars.len(), steps);

    let (sig, noise) = schedule.get_scales(0);
    // alpha_bar_0 = alpha_0 = 1 - beta_0
    // sig = sqrt(alpha_bar)
    // noise = sqrt(1-alpha_bar)
    assert_relative_eq!(
        sig.powi(2) + noise.powi(2),
        1.0,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
}

#[test]
#[verified_engine::verified]
fn test_inject_noise() {
    let rows = 4;
    let cols = 4;
    let image = DMatrix::from_element(rows, cols, 1.0);
    let noise = DMatrix::from_element(rows, cols, 0.5);
    let sig = 0.8;
    let noise_scale = 0.6;

    let noisy = inject_noise(&image, &noise, sig, noise_scale);

    // 1.0 * 0.8 + 0.5 * 0.6 = 0.8 + 0.3 = 1.1
    assert_relative_eq!(noisy[(0, 0)], 1.1);
}
