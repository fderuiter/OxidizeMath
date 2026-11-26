use super::stats::SortingResult;

/// Defines a strategy for sorting a collection of elements.
///
/// This trait allows for the interchangeability of sorting algorithms
/// within the system, adhering to the Open/Closed Principle.
pub trait Sorter<T> {
    /// Sorts the given data slice.
    ///
    /// # Arguments
    ///
    /// * `data` - The slice of data to be sorted.
    ///
    /// # Returns
    ///
    /// A `SortingResult` containing the sorted data and performance statistics.
    #[verified_engine::verified]
    fn sort(&self, data: &[T]) -> SortingResult<T>;
}
