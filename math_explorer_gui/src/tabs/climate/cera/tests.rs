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
