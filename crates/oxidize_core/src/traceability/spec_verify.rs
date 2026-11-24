//! Verification of `docs/math_specs/` files and `papers/references.bib` persistent identifiers.

use super::{TraceabilityEngine, TraceabilityReport};
use crate::bibtex::{BibtexDatabase, extract_tex_cite_keys, validate_math_spec_markdown};
use crate::vfs::VirtualFileSystem;
use std::collections::HashSet;

fn workspace_base_candidates(papers_dir: &str) -> Vec<String> {
    let norm_papers = crate::path_utils::normalize_path(papers_dir);
    let mut bases = Vec::new();
    if let Some((parent, _)) = norm_papers.trim_end_matches('/').rsplit_once('/')
        && !parent.is_empty()
    {
        bases.push(format!("{parent}/"));
    }
    bases.push(String::new());
    #[cfg(not(target_arch = "wasm32"))]
    {
        bases.push("../".to_string());
        bases.push("../../".to_string());
    }
    bases.dedup();
    bases
}

fn is_safe_spec_id(spec_id: &str) -> bool {
    !spec_id.is_empty()
        && !spec_id.contains("..")
        && spec_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
}

impl<V: VirtualFileSystem> TraceabilityEngine<V> {
    async fn read_workspace_rel_optional(
        &self,
        papers_dir: &str,
        rel_path: &str,
    ) -> Result<Option<String>, std::io::Error> {
        for base in workspace_base_candidates(papers_dir) {
            let candidate = format!("{base}{rel_path}");
            match self.read_optional_file(&candidate).await {
                Ok(Some(content)) => return Ok(Some(content)),
                Ok(None) => {}
                Err(err)
                    if base.starts_with("..")
                        && err.kind() == std::io::ErrorKind::PermissionDenied => {}
                Err(err) => return Err(err),
            }
        }
        Ok(None)
    }

    pub(super) async fn verify_specs_and_bibtex(
        &self,
        papers_dir: &str,
        valid_papers: &HashSet<String>,
        registry_table: &toml::Table,
        report: &mut TraceabilityReport,
    ) -> Result<HashSet<String>, std::io::Error> {
        let bib_rel = format!(
            "{}/references.bib",
            crate::path_utils::normalize_path(papers_dir).trim_end_matches('/')
        );
        let bib_raw = match self.read_optional_file(&bib_rel).await? {
            Some(raw) => Some(raw),
            None => {
                self.read_workspace_rel_optional(papers_dir, "papers/references.bib")
                    .await?
            }
        };
        let parsed_bib = match bib_raw {
            Some(raw) => match BibtexDatabase::parse(&raw) {
                Ok(db) => Some(db),
                Err(err) => {
                    report
                        .invalid_links
                        .push(("papers/references.bib".to_string(), err));
                    None
                }
            },
            None => None,
        };

        self.verify_tex_citations(papers_dir, valid_papers, parsed_bib.as_ref(), report)
            .await?;

        self.verify_registry_specs(papers_dir, registry_table, parsed_bib.as_ref(), report)
            .await
    }

    async fn verify_tex_citations(
        &self,
        papers_dir: &str,
        valid_papers: &HashSet<String>,
        parsed_bib: Option<&BibtexDatabase>,
        report: &mut TraceabilityReport,
    ) -> Result<(), std::io::Error> {
        let norm_papers = crate::path_utils::normalize_path(papers_dir);
        let mut sorted_papers: Vec<&String> = valid_papers.iter().collect();
        sorted_papers.sort();

        for paper in sorted_papers {
            let path = crate::path_utils::join_and_normalize(&norm_papers, paper);
            let Some(tex_src) = self.read_optional_file(&path).await? else {
                continue;
            };
            let cited_keys = extract_tex_cite_keys(&tex_src);
            if cited_keys.is_empty() {
                continue;
            }
            let Some(bib) = parsed_bib else {
                report.invalid_links.push((
                    format!("papers/{paper}"),
                    "missing `papers/references.bib` for cited BibTeX keys".to_string(),
                ));
                continue;
            };
            for key in cited_keys {
                if let Err(err) = bib.verify_cited_key(&key) {
                    report
                        .invalid_links
                        .push((format!("papers/{paper}"), format!("bibtex:{key} ({err})")));
                }
            }
        }
        Ok(())
    }

    async fn verify_registry_specs(
        &self,
        papers_dir: &str,
        registry_table: &toml::Table,
        parsed_bib: Option<&BibtexDatabase>,
        report: &mut TraceabilityReport,
    ) -> Result<HashSet<String>, std::io::Error> {
        let mut verified_specs = HashSet::new();
        let Some(links) = registry_table.get("links").and_then(|v| v.as_table()) else {
            return Ok(verified_specs);
        };

        for (module_name, target_val) in links {
            let Some(target) = target_val.as_str() else {
                continue;
            };
            let Some(spec_id) = target.strip_prefix("spec:") else {
                continue;
            };
            if !is_safe_spec_id(spec_id) {
                report.invalid_links.push((
                    module_name.clone(),
                    format!("{target} (invalid spec identifier)"),
                ));
                continue;
            }
            let rel_spec = format!("docs/math_specs/{spec_id}.md");
            let Some(spec_md) = self
                .read_workspace_rel_optional(papers_dir, &rel_spec)
                .await?
            else {
                report.invalid_links.push((
                    module_name.clone(),
                    format!("{target} (missing `{rel_spec}`)"),
                ));
                continue;
            };
            let Some(bib) = parsed_bib else {
                report.invalid_links.push((
                    module_name.clone(),
                    format!("{target} (missing `papers/references.bib`)"),
                ));
                continue;
            };
            match validate_math_spec_markdown(spec_id, &spec_md, bib) {
                Ok(_) => {
                    verified_specs.insert(target.to_string());
                }
                Err(err) => {
                    report
                        .invalid_links
                        .push((module_name.clone(), format!("{target} ({err})")));
                }
            }
        }

        Ok(verified_specs)
    }
}
