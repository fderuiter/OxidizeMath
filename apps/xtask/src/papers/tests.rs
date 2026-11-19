use super::*;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};

static TEST_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct TempOutputDir {
    path: PathBuf,
}

impl TempOutputDir {
    fn new(label: &str) -> Self {
        let id = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir_name = format!(
            "oxidizemath_papers_test_{}_{}_{}",
            std::process::id(),
            label,
            id
        );
        let path = env::temp_dir().join(dir_name);
        if path.exists() {
            let _ = fs::remove_dir_all(&path);
        }
        fs::create_dir_all(&path).expect("failed to create temp test dir");
        Self { path }
    }
}

impl Drop for TempOutputDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[test]
fn test_is_safe_bundle_filename_rejects_leading_dashes() {
    assert!(is_safe_bundle_filename("paper1.pdf"));
    assert!(is_safe_bundle_filename("index.html"));
    assert!(is_safe_bundle_filename("Verification_Certificate.json"));

    assert!(!is_safe_bundle_filename("-"));
    assert!(!is_safe_bundle_filename("-v.pdf"));
    assert!(!is_safe_bundle_filename("-index.html"));
    assert!(!is_safe_bundle_filename("--checkpoint=1.pdf"));
    assert!(!is_safe_bundle_filename(
        "--checkpoint-action=exec=touch CANARY_PWNED.pdf"
    ));
    assert!(!is_safe_bundle_filename("SHA256SUMS"));
    assert!(!is_safe_bundle_filename("notes.txt"));
}

#[test]
fn test_collect_bundle_artifacts_filters_dash_prefixed_files() {
    let temp = TempOutputDir::new("collect");
    fs::write(temp.path.join("algebra.pdf"), b"%PDF-1.4\n").unwrap();
    fs::write(temp.path.join("index.html"), b"<html></html>").unwrap();
    fs::write(temp.path.join("Verification_Certificate.json"), b"{}").unwrap();
    fs::write(temp.path.join("--checkpoint=1.pdf"), b"malicious").unwrap();
    fs::write(
        temp.path
            .join("--checkpoint-action=exec=touch CANARY_PWNED.pdf"),
        b"malicious",
    )
    .unwrap();
    fs::write(temp.path.join("-rf.pdf"), b"malicious").unwrap();

    let collected = collect_bundle_artifacts(&temp.path);
    assert_eq!(
        collected,
        vec![
            "Verification_Certificate.json".to_string(),
            "algebra.pdf".to_string(),
            "index.html".to_string(),
        ]
    );
}

#[test]
fn test_tar_and_checksum_args_include_end_of_options_delimiter() {
    let files = vec!["algebra.pdf".to_string(), "index.html".to_string()];
    let tar_args = build_tar_args(&files);
    assert_eq!(
        tar_args,
        vec![
            "-czf",
            "papers-bundle.tar.gz",
            "--",
            "algebra.pdf",
            "index.html"
        ]
    );

    let sha256_args = build_checksum_args("sha256sum", &files);
    assert_eq!(sha256_args, vec!["--", "algebra.pdf", "index.html"]);

    let shasum_args = build_checksum_args("shasum", &files);
    assert_eq!(
        shasum_args,
        vec!["-a", "256", "--", "algebra.pdf", "index.html"]
    );
}

#[test]
#[cfg(unix)]
fn test_package_release_bundle_prevents_option_injection() {
    let temp = TempOutputDir::new("bundle_exec");
    fs::write(temp.path.join("paper_alpha.pdf"), b"%PDF-1.4 sample\n").unwrap();
    fs::write(temp.path.join("index.html"), b"<!DOCTYPE html></html>").unwrap();
    fs::write(temp.path.join("Verification_Certificate.json"), b"{}").unwrap();

    // Plant GNU tar option-injection payloads in output_dir
    fs::write(temp.path.join("--checkpoint=1.pdf"), b"payload").unwrap();
    fs::write(
        temp.path
            .join("--checkpoint-action=exec=touch CANARY_PWNED.pdf"),
        b"payload",
    )
    .unwrap();

    package_release_bundle(&temp.path);

    // Verify injected command was NOT executed
    assert!(
        !temp.path.join("CANARY_PWNED.pdf").exists(),
        "Option injection payload must not execute during bundle creation"
    );

    // Verify archive exists and contains only legitimate artifacts
    let bundle_path = temp.path.join("papers-bundle.tar.gz");
    assert!(bundle_path.exists(), "Release bundle should be created");

    let tar_list = Command::new("tar")
        .current_dir(&temp.path)
        .args(["-tzf", "papers-bundle.tar.gz"])
        .output()
        .expect("failed to list tar contents");
    assert!(tar_list.status.success());
    let mut entries: Vec<String> = String::from_utf8_lossy(&tar_list.stdout)
        .lines()
        .map(|s| s.trim_start_matches("./").to_string())
        .collect();
    entries.sort();
    assert_eq!(
        entries,
        vec![
            "Verification_Certificate.json".to_string(),
            "index.html".to_string(),
            "paper_alpha.pdf".to_string(),
        ]
    );

    // Verify SHA256SUMS excludes dash-prefixed entries and includes valid artifacts
    let sums = fs::read_to_string(temp.path.join("SHA256SUMS")).expect("SHA256SUMS missing");
    assert!(sums.contains("papers-bundle.tar.gz"));
    assert!(sums.contains("paper_alpha.pdf"));
    assert!(sums.contains("index.html"));
    assert!(sums.contains("Verification_Certificate.json"));
    assert!(!sums.contains("--checkpoint"));
    assert!(!sums.contains("CANARY_PWNED.pdf"));
}

#[test]
fn test_encode_uri_component_rfc3986_and_special_chars() {
    let unreserved = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
    assert_eq!(encode_uri_component(unreserved), unreserved);
    assert_eq!(encode_uri_component(""), "");

    assert_eq!(
        encode_uri_component("a<b>\"c'&d#e?f%25 g"),
        "a%3Cb%3E%22c%27%26d%23e%3Ff%2525%20g"
    );
    assert_eq!(
        encode_uri_component("théorème_π"),
        "th%C3%A9or%C3%A8me_%CF%80"
    );
}

#[test]
fn test_render_paper_card_escapes_and_encodes_stems() {
    let safe_card = render_paper_card(
        "advanced_linear_algebra",
        "Linear Algebra",
        "OxidizeMath Project",
        "Matrix decompositions.",
    );
    assert!(safe_card.contains("<code>advanced_linear_algebra.pdf</code>"));
    assert!(safe_card.contains(r#"href="advanced_linear_algebra.pdf""#));

    let special_stem = "paper_<tag>&\"quote\"'single'#frag?query%25 space";
    let special_card = render_paper_card(
        special_stem,
        "Safe <Title>",
        "Author & Co",
        "Abstract \"text\".",
    );
    assert!(special_card.contains(
        "<code>paper_&lt;tag&gt;&amp;&quot;quote&quot;&#39;single&#39;#frag?query%25 space.pdf</code>"
    ));
    assert!(special_card.contains(
        r#"href="paper_%3Ctag%3E%26%22quote%22%27single%27%23frag%3Fquery%2525%20space.pdf""#
    ));
    assert!(!special_card.contains("<tag>"));
    assert!(!special_card.contains("<Title>"));

    let utf8_card = render_paper_card("mécanique_π", "Mécanique", "Auteur", "Résumé.");
    assert!(utf8_card.contains("<code>mécanique_π.pdf</code>"));
    assert!(utf8_card.contains(r#"href="m%C3%A9canique_%CF%80.pdf""#));
}

#[test]
fn test_generate_index_html_csp_and_stem_encoding() {
    let temp = TempOutputDir::new("index_html");
    let out_dir = temp.path.join("output");
    fs::create_dir_all(&out_dir).unwrap();

    let delim_tex = temp.path.join("paper_'&d#e%25 space.tex");
    fs::write(
        &delim_tex,
        "\\documentclass{article}\n\\begin{document}\nBody.\n\\end{document}\n",
    )
    .unwrap();

    let utf8_tex = temp.path.join("étude_π.tex");
    fs::write(
        &utf8_tex,
        "\\documentclass{article}\n\\title{Étude}\n\\begin{document}\n\\begin{abstract}UTF-8.\\end{abstract}\n\\end{document}\n",
    )
    .unwrap();

    let synthetic_html_stem = temp.path.join("synthetic_<a>\"b\"?q.tex");
    let tex_files = vec![delim_tex, utf8_tex, synthetic_html_stem];

    generate_index_html(&tex_files, &out_dir);

    let html = fs::read_to_string(out_dir.join("index.html")).expect("index.html missing");
    assert!(html.contains(
        r#"<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">"#
    ));
    assert!(html.contains("<code>paper_&#39;&amp;d#e%25 space.pdf</code>"));
    assert!(html.contains(r#"href="paper_%27%26d%23e%2525%20space.pdf""#));
    assert!(html.contains("<h2 class=\"card-title\">Paper &#39;&amp;d#e%25 Space</h2>"));
    assert!(html.contains("<code>étude_π.pdf</code>"));
    assert!(html.contains(r#"href="%C3%A9tude_%CF%80.pdf""#));
    assert!(html.contains("<code>synthetic_&lt;a&gt;&quot;b&quot;?q.pdf</code>"));
    assert!(html.contains(r#"href="synthetic_%3Ca%3E%22b%22%3Fq.pdf""#));
    assert!(!html.contains("<a>"));
}

#[test]
#[cfg(unix)]
fn test_generate_index_html_canary_tex_on_unix() {
    let temp = TempOutputDir::new("canary_html");
    let papers_dir = temp.path.join("papers");
    let out_dir = papers_dir.join("output");
    fs::create_dir_all(&out_dir).unwrap();

    let canary_tex = papers_dir.join("paper_<canary-marker>.tex");
    fs::write(
        &canary_tex,
        "\\documentclass{article}\n\\title{Safe <canary-title>}\n\\begin{document}\n\\begin{abstract}Canary.\\end{abstract}\nBody.\n\\end{document}\n",
    )
    .unwrap();

    generate_index_html(&[canary_tex], &out_dir);

    let html = fs::read_to_string(out_dir.join("index.html")).expect("index.html missing");
    assert!(html.contains("Safe &lt;canary-title&gt;"));
    assert!(!html.contains("<canary-marker>"));
    assert!(html.contains("<code>paper_&lt;canary-marker&gt;.pdf</code>"));
    assert!(html.contains(r#"href="paper_%3Ccanary-marker%3E.pdf""#));
}
