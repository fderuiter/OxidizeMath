//! Dedicated integration tests for typed `AIError` returns on `SculptedGridMdp::apply_cells_u8`.

use diagnostics::Diagnostic;
use domain_ai::ai::reinforcement_learning::maze::SculptedGridMdp;
use domain_ai::error::AIError;

#[test]
fn sculpted_grid_mdp_apply_cells_u8_returns_typed_ai_errors_and_diagnostics() {
    let mut mdp = SculptedGridMdp::default_with_traps(4, 3, 0.95);
    let len_err = mdp.apply_cells_u8(&[0; 5]).unwrap_err();
    assert_eq!(
        len_err,
        AIError::CellMaskLengthMismatch {
            expected: 12,
            actual: 5,
        }
    );
    assert_eq!(
        len_err.metadata().get("expected").map(String::as_str),
        Some("12")
    );

    let mut invalid_tag = mdp.cells_to_u8();
    invalid_tag[7] = 250;
    let tag_err = mdp.apply_cells_u8(&invalid_tag).unwrap_err();
    assert_eq!(
        tag_err,
        AIError::InvalidCellRoleTag {
            index: 7,
            code: 250,
        }
    );
    assert_eq!(
        tag_err.metadata().get("code").map(String::as_str),
        Some("250")
    );

    let valid = mdp.cells_to_u8();
    assert!(mdp.apply_cells_u8(&valid).is_ok());
}
