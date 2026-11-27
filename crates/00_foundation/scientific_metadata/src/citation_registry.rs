#![cfg_attr(any(), verified(opt_out = "metadata"))]

use std::collections::HashMap;
use std::fmt;
use std::sync::{OnceLock, RwLock};

/// Stack-allocated error returned when the global [`CitationRegistry`] lock is poisoned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CitationRegistryError {
    /// A thread panicked while holding the global citation registry `RwLock`.
    PoisonedLock,
}

impl fmt::Display for CitationRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PoisonedLock => write!(f, "Global CitationRegistry RwLock is poisoned"),
        }
    }
}

impl std::error::Error for CitationRegistryError {}

/// A thread-safe global registry for storing and retrieving academic citations.
///
/// `CitationRegistry` maps unique model or theory identifiers to their bibliographic
/// reference strings. This enables user interfaces and reporting tools to display
/// formal citations on demand without requiring mathematical simulation kernels
/// to maintain persistent registry state.
///
/// # Example
///
/// ```rust
/// use scientific_metadata::citation_registry::CitationRegistry;
///
/// CitationRegistry::register(
///     "ising_model".to_string(),
///     "Ising, E. (1925). Beitrag zur Theorie des Ferromagnetismus.".to_string(),
/// );
///
/// let citation = CitationRegistry::get("ising_model");
/// assert_eq!(
///     citation.as_deref(),
///     Some("Ising, E. (1925). Beitrag zur Theorie des Ferromagnetismus.")
/// );
/// ```
pub struct CitationRegistry {
    citations: HashMap<String, String>,
}

impl CitationRegistry {
    /// Returns a static reference to the global, `RwLock`-synchronized [`CitationRegistry`] singleton.
    ///
    /// The registry is lazily initialized on first access using [`OnceLock`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use scientific_metadata::citation_registry::CitationRegistry;
    ///
    /// let registry = CitationRegistry::global();
    /// let read_lock = registry.read().unwrap();
    /// ```
    pub fn global() -> &'static RwLock<CitationRegistry> {
        static REGISTRY: OnceLock<RwLock<CitationRegistry>> = OnceLock::new();
        REGISTRY.get_or_init(|| {
            RwLock::new(CitationRegistry {
                citations: HashMap::new(),
            })
        })
    }

    /// Fallibly registers an academic citation in the global registry without panicking on lock poison.
    pub fn try_register(id: String, citation: String) -> Result<(), CitationRegistryError> {
        Self::global()
            .write()
            .map_err(|_| CitationRegistryError::PoisonedLock)?
            .citations
            .insert(id, citation);
        Ok(())
    }

    /// Registers an academic citation for the specified identifier in the global registry.
    ///
    /// If an entry already exists for `id`, its citation text is overwritten.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the model, theory, or simulation regime.
    /// * `citation` - Bibliographic reference text (e.g., APA/IEEE citation or DOI).
    ///
    /// # Example
    ///
    /// ```rust
    /// use scientific_metadata::citation_registry::CitationRegistry;
    ///
    /// CitationRegistry::register(
    ///     "navier_stokes".to_string(),
    ///     "Navier, C.-L. (1822). Mémoire sur les lois du mouvement des fluides.".to_string(),
    /// );
    /// ```
    pub fn register(id: String, citation: String) {
        Self::try_register(id, citation).expect("Global CitationRegistry write lock poisoned");
    }

    /// Fallibly retrieves the academic citation associated with `id` without panicking on lock poison.
    pub fn try_get(id: &str) -> Result<Option<String>, CitationRegistryError> {
        Ok(Self::global()
            .read()
            .map_err(|_| CitationRegistryError::PoisonedLock)?
            .citations
            .get(id)
            .cloned())
    }

    /// Retrieves the academic citation associated with the specified identifier, if registered.
    ///
    /// Returns `Some(String)` containing the citation text if present, or `None` if no
    /// citation has been registered for `id`.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the model or theory to query.
    ///
    /// # Example
    ///
    /// ```rust
    /// use scientific_metadata::citation_registry::CitationRegistry;
    ///
    /// CitationRegistry::register("hopfield".to_string(), "Hopfield, J. J. (1982).".to_string());
    /// assert_eq!(CitationRegistry::get("hopfield").as_deref(), Some("Hopfield, J. J. (1982)."));
    /// assert_eq!(CitationRegistry::get("non_existent_model"), None);
    /// ```
    pub fn get(id: &str) -> Option<String> {
        Self::try_get(id).expect("Global CitationRegistry read lock poisoned")
    }
}
