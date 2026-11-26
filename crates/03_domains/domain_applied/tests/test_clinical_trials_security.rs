//! Integration tests for test clinical trials security.
use domain_applied::applied::clinical_trials::types::{ClinicalTrialError, GroupData};

#[test]
#[verified_engine::verified]
fn test_group_data_nan_rejection() {
    let data = vec![1.0, 2.0, f64::NAN, 4.0];

    // Sentinel: Verify that NaN is rejected.
    match GroupData::new(data) {
        Ok(_) => panic!("GroupData::new should reject NaN values"),
        Err(ClinicalTrialError::InvalidData(msg)) => {
            assert!(msg.contains("Data contains non-finite value"));
        }
        Err(e) => panic!("Expected InvalidData error, got {:?}", e),
    }
}

#[test]
#[verified_engine::verified]
fn test_group_data_infinite_rejection() {
    let data = vec![1.0, 2.0, f64::INFINITY, 4.0];

    // Sentinel: Verify that Infinity is rejected.
    match GroupData::new(data) {
        Ok(_) => panic!("GroupData::new should reject Infinite values"),
        Err(ClinicalTrialError::InvalidData(msg)) => {
            assert!(msg.contains("Data contains non-finite value"));
        }
        Err(e) => panic!("Expected InvalidData error, got {:?}", e),
    }
}
