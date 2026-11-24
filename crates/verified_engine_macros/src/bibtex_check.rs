use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use syn::LitStr;

#[path = "../../oxidize_core/src/bibtex/parser.rs"]
mod parser;

use parser::BibtexDatabase;

type BibCache = Mutex<HashMap<PathBuf, Result<Arc<BibtexDatabase>, String>>>;

fn bib_cache() -> &'static BibCache {
    static CACHE: OnceLock<BibCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn load_workspace_bib(lit: &LitStr) -> syn::Result<Arc<BibtexDatabase>> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR").map(PathBuf::from);
    let bib_lit = LitStr::new("papers/references.bib", lit.span());
    let (bib_path, _) =
        crate::embed_theory::resolve_tex_path_from_manifest(&bib_lit, manifest_dir.as_deref())
            .map_err(|_| {
                syn::Error::new_spanned(
                    lit,
                    "Could not locate `papers/references.bib` from `CARGO_MANIFEST_DIR` while verifying `#[theory(citation = ...)]`",
                )
            })?;

    let mut guard = bib_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let cached = guard
        .entry(bib_path.clone())
        .or_insert_with(|| {
            let content = fs::read_to_string(&bib_path)
                .map_err(|e| format!("Failed to read `{}`: {e}", bib_path.display()))?;
            BibtexDatabase::parse(&content).map(Arc::new)
        })
        .clone();

    cached.map_err(|msg| syn::Error::new_spanned(lit, msg))
}

pub(crate) fn validate_theory_citation_lit(lit: &LitStr) -> syn::Result<String> {
    let raw = lit.value();
    let keys: Vec<&str> = raw
        .split([',', ';'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if keys.is_empty() {
        return Err(syn::Error::new_spanned(
            lit,
            "`#[theory(citation = ...)]` must specify at least one verified BibTeX key from `papers/references.bib`",
        ));
    }

    let db = load_workspace_bib(lit)?;
    for key in keys {
        db.verify_cited_key(key)
            .map_err(|msg| syn::Error::new_spanned(lit, msg))?;
    }
    Ok(raw)
}
