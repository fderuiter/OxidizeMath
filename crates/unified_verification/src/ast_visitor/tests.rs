use super::*;

fn ast_flags(code: &str) -> (bool, bool, bool) {
    let file = syn::parse_file(code).unwrap();
    let mut visitor = MaliciousCodeVisitor::default();
    visitor.visit_file(&file);
    (
        visitor.has_shell_execution,
        visitor.has_network_socket,
        visitor.has_suspicious_fs,
    )
}

#[test]
fn test_clone_detector_is_a_b_plus_c() {
    let code = "fn test() { let x = a * b + c; }";
    let file = syn::parse_file(code).unwrap();
    if let syn::Item::Fn(f) = &file.items[0]
        && let syn::Stmt::Local(l) = &f.block.stmts[0]
        && let Some(init) = &l.init
    {
        assert!(CloneDetectorVisitor::is_a_b_plus_c(&init.expr));
    }
}

#[test]
fn test_run_clone_detector_execution() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    assert!(
        run_clone_detector(),
        "First-party code should pass clone detector"
    );
}

#[test]
fn test_prefilter_rejects_clean_and_unrelated_paths() {
    for snippet in [
        "fn add(a: i32, b: i32) -> i32 { a + b }",
        "fn benign(c: &mut std::process::Command) { let _ = std::process::Command::output(c); }",
        "fn net(s: &std::net::TcpStream, u: &std::net::UdpSocket) { let _ = (std::net::TcpStream::peer_addr(s), std::net::UdpSocket::local_addr(u)); }",
    ] {
        assert!(
            !may_contain_malicious_call(snippet),
            "unexpected match: {snippet}"
        );
        assert_eq!(ast_flags(snippet), (false, false, false));
    }
}

#[test]
fn test_prefilter_accepts_spaced_commented_raw_and_turbofish_calls() {
    for (snippet, expected) in [
        ("fn f() { Command :: new(\"ls\"); }", (true, false, false)),
        (
            "fn f() { TcpStream\n::\nconnect(\"a\"); }",
            (false, true, false),
        ),
        (
            "fn f() { UdpSocket\t::\tbind(\"b\"); }",
            (false, true, false),
        ),
        (
            "fn f() { std::fs::remove_dir_all(\"x\"); }",
            (false, false, true),
        ),
        (
            "fn f() { Command::/* c */new(\"sh\"); }",
            (true, false, false),
        ),
        (
            "fn f() { TcpStream::// c\nconnect(\"x\"); }",
            (false, true, false),
        ),
        (
            "fn f() { UdpSocket /* c */ ::bind(\"x\"); }",
            (false, true, false),
        ),
        ("fn f() { r#Command::new(\"sh\"); }", (true, false, false)),
        ("fn f() { Command::r#new(\"sh\"); }", (false, false, false)),
        ("fn f() { Command::<u8>::new(); }", (true, false, false)),
        (
            "fn f() { TcpStream::<u8>::connect(); }",
            (false, true, false),
        ),
        ("fn f() { UdpSocket::<u8>::bind(); }", (false, true, false)),
    ] {
        assert!(may_contain_malicious_call(snippet), "missed: {snippet}");
        assert_eq!(ast_flags(snippet), expected, "AST mismatch: {snippet}");
    }
}

#[test]
fn test_target_resolution_and_metadata_extraction() {
    let rustc_vv = "rustc 1.99.0\nhost: x86_64-unknown-linux-gnu\n";
    assert_eq!(
        resolve_target_triple(Some("wasm32-unknown-unknown"), Some(rustc_vv)),
        Some("wasm32-unknown-unknown".to_string())
    );
    assert_eq!(
        resolve_target_triple(Some("   "), Some(rustc_vv)),
        Some("x86_64-unknown-linux-gnu".to_string())
    );
    assert_eq!(resolve_target_triple(Some(""), None), None);

    let meta = serde_json::json!({
        "packages": [
            {"manifest_path": "/cache/serde-1.0/Cargo.toml", "source": "registry+https://github.com/rust-lang/crates.io-index"},
            {"manifest_path": "/app/crates/pure_math/Cargo.toml", "source": null}
        ]
    });
    assert_eq!(
        extract_crates_io_dep_paths(&meta),
        vec![PathBuf::from("/cache/serde-1.0")]
    );
}

#[test]
fn test_run_ast_visitor_execution() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    assert!(run_ast_visitor());
}
