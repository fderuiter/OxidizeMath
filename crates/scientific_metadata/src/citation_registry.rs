#![cfg_attr(any(), verified(opt_out = "metadata"))]

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

#[allow(missing_docs)]
pub struct CitationRegistry {
    citations: HashMap<String, String>,
}

impl CitationRegistry {
    #[allow(missing_docs)]
    pub fn global() -> &'static RwLock<CitationRegistry> {
        static REGISTRY: OnceLock<RwLock<CitationRegistry>> = OnceLock::new();
        REGISTRY.get_or_init(|| {
            RwLock::new(CitationRegistry {
                citations: HashMap::new(),
            })
        })
    }

    #[allow(missing_docs)]
    pub fn register(id: String, citation: String) {
        Self::global()
            .write()
            .unwrap()
            .citations
            .insert(id, citation);
    }

    #[allow(missing_docs)]
    pub fn get(id: &str) -> Option<String> {
        Self::global().read().unwrap().citations.get(id).cloned()
    }
}
