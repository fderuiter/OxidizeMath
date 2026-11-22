use super::{FavoritismInputs, calculate_favoritism_score};

/// Represents a child with a name and a set of attributes for favoritism calculation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Child {
    #[allow(missing_docs)]
    pub name: String,
    #[allow(missing_docs)]
    pub inputs: FavoritismInputs,
}

/// Determines the favorite child from a list of children based on the favoritism score.
///
/// This function iterates through a slice of `Child` structs, calculates the favoritism
/// score for each, and returns the child with the highest score.
///
/// # Arguments
///
/// * `children` - A slice of `Child` structs.
///
/// # Returns
///
/// An `Option` containing a reference to the `Child` with the highest favoritism score.
/// Returns `None` if the slice is empty.
#[verified_engine::verified]
pub fn find_favorite_child(children: &[Child]) -> Option<&Child> {
    children.iter().max_by(|a, b| {
        let score_a = calculate_favoritism_score(&a.inputs);
        let score_b = calculate_favoritism_score(&b.inputs);
        score_a
            .partial_cmp(&score_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}
