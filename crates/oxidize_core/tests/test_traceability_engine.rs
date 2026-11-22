//! Tests for oxidize_core TraceabilityEngine.

#![cfg(not(target_arch = "wasm32"))]

use oxidize_core::traceability::{AstCache, TraceabilityEngine, TraceabilityReport};
use oxidize_core::vfs::{DefaultVfs, VirtualFileSystem};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

struct MockVfs {
    dirs: HashMap<String, Vec<String>>,
    files: HashMap<String, Result<String, std::io::ErrorKind>>,
}

impl VirtualFileSystem for MockVfs {
    fn read_to_string(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String, std::io::Error>> + '_>> {
        let key = path.to_string();
        Box::pin(async move {
            match self.files.get(&key) {
                Some(Ok(content)) => Ok(content.clone()),
                Some(Err(kind)) => Err(std::io::Error::new(*kind, format!("Failed reading {key}"))),
                None => Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Missing file {key}"),
                )),
            }
        })
    }

    fn write_to_file(
        &self,
        _path: &str,
        _content: &[u8],
    ) -> Pin<Box<dyn Future<Output = Result<(), std::io::Error>> + '_>> {
        Box::pin(async move { Ok(()) })
    }

    fn list_dir(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, std::io::Error>> + '_>> {
        let key = path.to_string();
        Box::pin(async move {
            self.dirs.get(&key).cloned().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, format!("Missing dir {key}"))
            })
        })
    }
}

#[test]
fn test_traceability_engine_scan() {
    let (is_root, _) = oxidize_core::workspace::get_workspace_members();
    let prefix = if is_root { "" } else { "../../" };
    let engine = TraceabilityEngine::new(DefaultVfs);
    let d = format!("{prefix}crates/pure_math/src");
    let dirs_ref = [d.as_str()];
    let papers_dir = format!("{prefix}papers");
    let res = futures::executor::block_on(engine.scan_repository(&dirs_ref, &papers_dir, false));
    assert!(res.is_ok(), "Traceability scan of pure_math should succeed");
    let report = res.unwrap();
    assert!(report.scanned_files > 0);
    assert!(report.orphaned_papers.is_empty());

    let serialized = toml::to_string(&report).expect("serialize TraceabilityReport");
    let roundtrip: TraceabilityReport =
        toml::from_str(&serialized).expect("deserialize TraceabilityReport");
    assert_eq!(roundtrip, report);
}

#[test]
fn test_scan_repository_reports_error_when_registry_missing() {
    let mut dirs = HashMap::new();
    dirs.insert("papers".to_string(), vec!["essay.tex".to_string()]);
    let vfs = MockVfs {
        dirs,
        files: HashMap::new(),
    };
    let engine = TraceabilityEngine::new(vfs);
    let res = futures::executor::block_on(engine.scan_repository(&["src"], "papers", false));
    assert!(
        res.is_err(),
        "Missing traceability.toml must return Err instead of orphaned papers"
    );
}

#[test]
fn test_scan_repository_reports_error_when_discovered_source_unreadable() {
    let mut dirs = HashMap::new();
    dirs.insert("papers".to_string(), vec!["essay.tex".to_string()]);
    dirs.insert("src".to_string(), vec!["broken.rs".to_string()]);

    let mut files = HashMap::new();
    files.insert(
        "traceability.toml".to_string(),
        Ok("[links]\nessay = \"essay.tex\"\n".to_string()),
    );
    files.insert(
        "src/broken.rs".to_string(),
        Err(std::io::ErrorKind::NotFound),
    );

    let engine = TraceabilityEngine::new(MockVfs { dirs, files });
    let res = futures::executor::block_on(engine.scan_repository(&["src"], "papers", false));
    assert!(
        res.is_err(),
        "Unreadable discovered .rs file must return Err instead of silent partial scan"
    );
}

#[test]
fn test_scan_repository_reports_error_when_papers_dir_missing() {
    let mut files = HashMap::new();
    files.insert(
        "traceability.toml".to_string(),
        Ok("[links]\nessay = \"essay.tex\"\n".to_string()),
    );
    let engine = TraceabilityEngine::new(MockVfs {
        dirs: HashMap::new(),
        files,
    });
    let res =
        futures::executor::block_on(engine.scan_repository(&["src"], "missing_papers", false));
    assert!(res.is_err(), "Missing papers_dir must return Err");
}

#[test]
fn test_scan_repository_with_cache_parses_each_file_at_most_once() {
    let (is_root, _) = oxidize_core::workspace::get_workspace_members();
    let prefix = if is_root { "" } else { "../../" };
    let engine = TraceabilityEngine::new(DefaultVfs);
    let d = format!("{prefix}crates/pure_math/src");
    let dirs_ref = [d.as_str()];
    let papers_dir = format!("{prefix}papers");

    let mut cache = AstCache::new();
    let first = futures::executor::block_on(engine.scan_repository_with_cache(
        &dirs_ref,
        &papers_dir,
        false,
        &mut cache,
    ))
    .expect("first scan with cache should succeed");

    assert!(first.scanned_files > 0);
    assert_eq!(
        cache.parse_calls(),
        first.scanned_files,
        "Each active Rust file in pure_math must be parsed exactly once instead of twice"
    );

    let parse_calls_after_first = cache.parse_calls();
    let second = futures::executor::block_on(engine.scan_repository_with_cache(
        &dirs_ref,
        &papers_dir,
        false,
        &mut cache,
    ))
    .expect("second scan reusing warm cache should succeed");

    assert_eq!(second, first);
    assert_eq!(
        cache.parse_calls(),
        parse_calls_after_first,
        "Warm AstCache must reuse parsed ASTs with zero additional syn::parse_file calls"
    );
}
