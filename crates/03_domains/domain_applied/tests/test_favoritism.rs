//! Integration tests for test favoritism.
use domain_applied::applied::favoritism::{FavoritismInputs, calculate_favoritism_score};

#[test]
#[verified_engine::verified]
fn test_favoritism_defaults() {
    let inputs = FavoritismInputs::default();
    let score = calculate_favoritism_score(&inputs);
    // Since there is a random factor 0.9..1.1, the score will vary, but should be positive.
    assert!(score > 0.0);
    assert!(score.is_finite());
}
