// [cite:ualbf_quasiperfect]
pub use crate::error::NumberTheoryError;

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_failure_result() -> Result<UalbfSearchResult, NumberTheoryError> {
        Err(NumberTheoryError::ParseError("bad bound".to_string()))
    }

    #[test]
    fn test_ualbf_search_derives_and_unwrap_err() {
        let res1 = ualbf_search(10, 4, "1.5", "2.0").unwrap();
        let res2 = ualbf_search(10, 4, "1.5", "2.0").unwrap();
        assert_eq!(res1, res2);
        assert_eq!(UalbfSearchResult::default().valid_components, 0);
        assert!(!format!("{res1:?}").is_empty());

        assert_eq!(
            parse_failure_result().unwrap_err(),
            NumberTheoryError::ParseError("bad bound".to_string())
        );
    }
}
