use nalgebra::DMatrix;
use std::collections::HashMap;

/// Represents a LoRA state dictionary as a map from tensor names to matrices.
///
/// Encapsulated to prevent exposing the internal `HashMap` structure directly,
/// adhering to the Newtype pattern for robustness and encapsulation.
#[derive(Debug, Clone, Default)]
pub struct LoraStateDict {
    tensors: HashMap<String, DMatrix<f64>>,
}

impl LoraStateDict {
    /// Creates a new, empty `LoraStateDict`.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }

    /// Inserts a tensor into the state dictionary.
    #[verified_engine::verified]
    pub fn insert(&mut self, key: String, tensor: DMatrix<f64>) {
        self.tensors.insert(key, tensor);
    }

    /// Gets a reference to a tensor from the state dictionary.
    #[verified_engine::verified]
    pub fn get(&self, key: &str) -> Option<&DMatrix<f64>> {
        self.tensors.get(key)
    }

    /// Returns an iterator over the key-value pairs in the dictionary.
    #[verified_engine::verified]
    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, String, DMatrix<f64>> {
        self.tensors.iter()
    }

    /// Returns a mutable iterator over the key-value pairs in the dictionary.
    #[verified_engine::verified]
    pub fn iter_mut(&mut self) -> std::collections::hash_map::IterMut<'_, String, DMatrix<f64>> {
        self.tensors.iter_mut()
    }
}

impl std::iter::FromIterator<(String, DMatrix<f64>)> for LoraStateDict {
    fn from_iter<I: IntoIterator<Item = (String, DMatrix<f64>)>>(iter: I) -> Self {
        Self {
            tensors: iter.into_iter().collect(),
        }
    }
}

impl<'a> IntoIterator for &'a LoraStateDict {
    type Item = (&'a String, &'a DMatrix<f64>);
    type IntoIter = std::collections::hash_map::Iter<'a, String, DMatrix<f64>>;

    #[verified_engine::verified]
    fn into_iter(self) -> Self::IntoIter {
        self.tensors.iter()
    }
}

impl<'a> IntoIterator for &'a mut LoraStateDict {
    type Item = (&'a String, &'a mut DMatrix<f64>);
    type IntoIter = std::collections::hash_map::IterMut<'a, String, DMatrix<f64>>;

    #[verified_engine::verified]
    fn into_iter(self) -> Self::IntoIter {
        self.tensors.iter_mut()
    }
}

impl IntoIterator for LoraStateDict {
    type Item = (String, DMatrix<f64>);
    type IntoIter = std::collections::hash_map::IntoIter<String, DMatrix<f64>>;

    #[verified_engine::verified]
    fn into_iter(self) -> Self::IntoIter {
        self.tensors.into_iter()
    }
}
