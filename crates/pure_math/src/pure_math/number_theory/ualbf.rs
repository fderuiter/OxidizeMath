// [cite:ualbf_quasiperfect]
use thiserror::Error;

#[derive(Error, Debug)]
#[allow(missing_docs)]
pub enum NumberTheoryError {
    #[error("Parse error: {0}")]
    #[allow(missing_docs)]
    ParseError(String),
}

#[allow(missing_docs)]
pub struct UalbfSearchResult {
    #[allow(missing_docs)]
    pub message: String,
    #[allow(missing_docs)]
    pub valid_components: usize,
    #[allow(missing_docs)]
    pub pruned_components: usize,
    #[allow(missing_docs)]
    pub prefix_count: usize,
    #[allow(missing_docs)]
    pub rejected_by_lattice: usize,
    #[allow(missing_docs)]
    pub candidates_checked: usize,
}

#[allow(missing_docs)]
#[verified_engine::verified]
pub fn ualbf_search(
    _limit_p: u64,
    _max_exponent: u32,
    _stop_threshold_str: &str,
    _target_max_str: &str,
) -> Result<UalbfSearchResult, NumberTheoryError> {
    Ok(UalbfSearchResult {
        message: "Not supported on WASM due to GMP limitation".to_string(),
        valid_components: 0,
        pruned_components: 0,
        prefix_count: 0,
        rejected_by_lattice: 0,
        candidates_checked: 0,
    })
}
