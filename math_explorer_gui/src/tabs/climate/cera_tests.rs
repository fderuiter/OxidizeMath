#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_cera_tool_reduces_loss_and_emits_distinct_channels() {
    let mut tool = CeraTool::default();
    tool.train_epochs(3);
    tool.train_epochs(3);

    assert_eq!(tool.epochs_trained, 6);
    assert_eq!(tool.loss_history.len(), 6);
    let first_loss = *tool.loss_history.first().unwrap();
    let last_loss = *tool.loss_history.last().unwrap();
    assert!(
        last_loss < first_loss,
        "Expected CERA loss to decrease across epochs ({first_loss} -> {last_loss})"
    );

    let preds = tool
        .prediction_result
        .as_ref()
        .expect("prediction populated")
        .as_ref()
        .expect("valid prediction");
    assert_eq!(preds.len(), tool.output_size);
    assert!(
        preds.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-6),
        "Expected distinct channel outputs, got {preds:?}"
    );
}

#[test]
fn test_cera_tool_invalid_alignment_emits_climate_error_telemetry() {
    let _telemetry_guard = crate::reflective_ui::tests::lock_test_mutex();
    let _ = crate::telemetry::poll_app_telemetry();

    let mut tool = CeraTool {
        latent_channels: 2,
        aligned_channels: 4,
        ..Default::default()
    };
    tool.train_epochs(1);

    assert!(matches!(tool.prediction_result, Some(Err(_))));
    let events = crate::telemetry::poll_app_telemetry();
    assert!(events.iter().any(|ev| {
        ev.source == "CeraEmulator"
            && ev.metadata.get("umbrella_type").map(String::as_str) == Some("ClimateError")
    }));
}
