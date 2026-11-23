//! Per-file AST and citation scanning helpers for [`super::TraceabilityEngine`].

use super::analysis::{
    analyze_parsed_ast, collect_valid_citations, extract_citations_with_lines,
    fix_and_collect_citations,
};
use super::{AstCache, TraceabilityEngine, TraceabilityReport};
use crate::vfs::VirtualFileSystem;
use std::collections::{HashMap, HashSet};

impl<V: VirtualFileSystem> TraceabilityEngine<V> {
    pub(super) async fn process_code_files(
        &self,
        code_files: Vec<String>,
        registered_modules: &HashSet<String>,
        registry_links: &HashMap<String, String>,
        report: &mut TraceabilityReport,
        auto_fix: bool,
        cache: &mut AstCache,
    ) -> Result<(), std::io::Error> {
        for file in code_files {
            self.process_single_file(
                file,
                registered_modules,
                registry_links,
                report,
                auto_fix,
                cache,
            )
            .await?;
        }
        Ok(())
    }

    async fn process_single_file(
        &self,
        file: String,
        registered_modules: &HashSet<String>,
        registry_links: &HashMap<String, String>,
        report: &mut TraceabilityReport,
        auto_fix: bool,
        cache: &mut AstCache,
    ) -> Result<(), std::io::Error> {
        let norm_file = crate::path_utils::normalize_path(&file);
        let key = AstCache::normalize_key(&norm_file);
        if cache.get(&key).is_none() {
            let content = self.vfs.read_to_string(&norm_file).await?;
            cache.insert_source(key.clone(), content);
        }

        report.scanned_files += 1;
        let is_module = norm_file.ends_with("mod.rs")
            || norm_file.contains("/tabs/")
            || norm_file.ends_with("lib.rs");

        let citations = cache
            .get(&key)
            .map(|entry| extract_citations_with_lines(&entry.content))
            .unwrap_or_default();

        let final_citations = if auto_fix {
            let mut updated = cache
                .get(&key)
                .map(|entry| entry.content.clone())
                .unwrap_or_default();
            let (file_modified, resolved) = fix_and_collect_citations(
                &norm_file,
                &mut updated,
                citations,
                registered_modules,
                registry_links,
                report,
                true,
            );
            if file_modified {
                self.vfs
                    .write_to_file(&norm_file, updated.as_bytes())
                    .await?;
                cache.insert_source(key.clone(), updated);
            }
            resolved
        } else {
            collect_valid_citations(&norm_file, citations, registered_modules, report)
        };

        let ast = cache.get(&key).and_then(|entry| entry.ast.as_ref());
        analyze_parsed_ast(
            &norm_file,
            ast,
            is_module,
            &final_citations,
            registered_modules,
            report,
        );
        Ok(())
    }
}
