//! Tier 5 security tests for `domain_climate`: `checked_mul` overflow guards,
//! zero-level/channel validation, subnormal stability, and zero-allocation CERA
//! inference and training via pre-allocated scratch workspaces.

use domain_climate::climate::cera::Cera;
use domain_climate::climate::config::CeraConfig;
use domain_climate::climate::loss::{
    EmdScratch, earth_movers_distance_into, try_earth_movers_distance_into, try_mse_loss,
};
use domain_climate::climate::scratch::{CeraScratch, CeraTrainingScratch};
use domain_climate::climate::training::CeraTrainer;
use domain_climate::error::ClimateError;
use nalgebra::DMatrix;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static TEST_ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

fn sample_config() -> CeraConfig {
    CeraConfig {
        learning_rate: 1e-3,
        lambda_pred: 0.1,
        lambda_emd: 0.05,
        epochs: 2,
        batch_size: 2,
        in_channels: 2,
        latent_channels: 3,
        aligned_channels: 2,
        num_levels: 4,
        output_size: 2,
    }
}

#[test]
fn cera_config_and_scratch_reject_zero_dimensions_and_checked_mul_overflow() {
    let mut zero_levels = sample_config();
    zero_levels.num_levels = 0;
    assert_eq!(Cera::new(zero_levels).err(), Some(ClimateError::ZeroLevels));

    let mut zero_aligned = sample_config();
    zero_aligned.aligned_channels = 0;
    assert_eq!(
        Cera::new(zero_aligned).err(),
        Some(ClimateError::ZeroAlignedChannels)
    );

    let mut misaligned = sample_config();
    misaligned.aligned_channels = 4;
    misaligned.latent_channels = 3;
    assert!(matches!(
        Cera::new(misaligned),
        Err(ClimateError::ChannelAlignmentExceeded { .. })
    ));

    let mut ovf_cfg = sample_config();
    ovf_cfg.num_levels = usize::MAX;
    ovf_cfg.aligned_channels = 2;
    ovf_cfg.latent_channels = 2;
    assert!(matches!(
        Cera::new(ovf_cfg.clone()),
        Err(ClimateError::DimensionOverflow { .. })
    ));
    assert!(matches!(
        CeraScratch::from_config(&ovf_cfg),
        Err(ClimateError::DimensionOverflow { .. })
    ));
    assert!(matches!(
        CeraTrainingScratch::from_config(&ovf_cfg),
        Err(ClimateError::DimensionOverflow { .. })
    ));
    assert!(matches!(
        CeraScratch::new(&sample_config(), usize::MAX),
        Err(ClimateError::DimensionOverflow { .. })
    ));
}

#[test]
fn cera_inference_emd_and_training_execute_with_zero_heap_allocations_and_subnormal_inputs() {
    let cfg = sample_config();
    let mut model = Cera::new(cfg.clone()).expect("valid Cera model");
    let mut infer_scratch = CeraScratch::from_config(&cfg).expect("CeraScratch");
    let mut train_scratch = CeraTrainingScratch::from_config(&cfg).expect("CeraTrainingScratch");
    let mut emd_scratch = EmdScratch::new(8);

    let sub = f32::MIN_POSITIVE * 0.5;
    assert!(sub > 0.0 && sub < f32::MIN_POSITIVE);

    let batch_rows = cfg.batch_size * cfg.num_levels;
    let control_inputs = DMatrix::from_fn(batch_rows * 2, cfg.in_channels, |r, c| {
        sub + 0.05 * ((r + c) as f32)
    });
    let warm_inputs = DMatrix::from_fn(batch_rows * 2, cfg.in_channels, |r, c| {
        sub * 2.0 + 0.06 * ((r + 2 * c) as f32)
    });
    let control_targets = DMatrix::from_fn(cfg.batch_size * 2, cfg.output_size, |r, c| {
        0.1 * ((r + c) as f32)
    });
    let single_batch = control_inputs.rows(0, batch_rows).clone_owned();
    let mut pred_out = DMatrix::zeros(cfg.batch_size, cfg.output_size);
    let mut epoch_losses = [0.0_f32; 2];

    let z1 = DMatrix::from_element(8, 2, sub);
    let z2 = DMatrix::from_element(8, 2, sub * 4.0);
    let mismatched = DMatrix::zeros(7, 2);
    assert!(matches!(
        try_mse_loss(&z1, &mismatched),
        Err(ClimateError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        try_earth_movers_distance_into(&z1, &mismatched, &mut emd_scratch),
        Err(ClimateError::ShapeMismatch { .. })
    ));

    let ((emd_val, epochs_recorded), metrics) = VerifiedEngine::run_verified(|| {
        model
            .predict_into(&single_batch, &mut infer_scratch, &mut pred_out)
            .expect("zero-alloc predict_into");
        let emd_val = earth_movers_distance_into(&z1, &z2, &mut emd_scratch);
        let mut trainer = CeraTrainer::new(&mut model);
        let epochs_recorded = trainer
            .train_into(
                &control_inputs,
                &control_targets,
                &warm_inputs,
                &mut train_scratch,
                &mut epoch_losses,
            )
            .expect("zero-alloc train_into");
        (emd_val, epochs_recorded)
    });

    assert_eq!(metrics.allocations, 0);
    assert_eq!(metrics.allocated_bytes, 0);
    assert_eq!(epochs_recorded, 2);
    assert!(emd_val.is_finite() && emd_val >= 0.0);
    assert!(pred_out.iter().all(|v| v.is_finite()));
    assert!(
        epoch_losses
            .iter()
            .all(|loss| loss.is_finite() && *loss >= 0.0)
    );
}
