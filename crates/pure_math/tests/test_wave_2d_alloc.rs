//! Integration tests enforcing zero dynamic heap allocation in `WaveField2D` under `VerifiedAllocator`.

use pure_math::pure_math::analysis::pde::wave::{
    WaveField2D, WaveFieldConfig2D, WaveSource2D, WaveSourceKind,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_wave_field_2d_verified_allocator_lock() {
    let config = WaveFieldConfig2D {
        width: 96,
        height: 96,
        dx: 1.0 / 96.0,
        dy: 1.0 / 96.0,
        ..WaveFieldConfig2D::default()
    };
    let mut field = WaveField2D::new(config);
    let src = WaveSource2D {
        kind: WaveSourceKind::PlaneWave {
            x_index: 14,
            y_min: 18,
            y_max: 78,
        },
        frequency: 12.0,
        amplitude: 50.0,
        phase: 0.0,
    };
    let dt = field.cfl_dt();

    let (ok_pos, _) = VerifiedEngine::run_verified(|| {
        for _ in 0..32 {
            field.step(dt, Some(&src));
        }
        field.is_finite()
    });
    assert!(ok_pos);

    field.clear_state();
    field.set_reference_frequency(12.0);
    field.configure_medium_in_place(|x, _y, _cfg, n_out, obs_out| {
        *obs_out = 0;
        *n_out = if (36..=56).contains(&x) { -1.0 } else { 1.0 };
    });
    let dt_neg = field.cfl_dt();
    let (ok_neg, _) = VerifiedEngine::run_verified(|| {
        for _ in 0..32 {
            field.step(dt_neg, Some(&src));
        }
        field.is_finite()
    });
    assert!(ok_neg);
}
