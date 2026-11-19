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

fn command_args_strings(cmd: &Command) -> Vec<String> {
    cmd.get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect()
}

fn command_env_value(cmd: &Command, key: &str) -> Option<String> {
    cmd.get_envs().find_map(|(k, v)| {
        if k == key {
            v.map(|val| val.to_string_lossy().into_owned())
        } else {
            None
        }
    })
}

#[test]
fn test_build_pdflatex_command_hardens_flags_and_kpathsea_env() {
    let tex_path = Path::new("papers/advanced_linear_algebra.tex");
    let out_dir = Path::new("papers/output");
    let cmd = build_pdflatex_command(tex_path, out_dir).expect("expected valid pdflatex command");

    assert_eq!(cmd.get_program(), "pdflatex");
    assert_eq!(
        command_args_strings(&cmd),
        vec![
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            "papers/output",
            "papers/advanced_linear_algebra.tex",
        ]
    );
    assert_eq!(
        command_env_value(&cmd, "TEXINPUTS").as_deref(),
        Some("papers//:")
    );
    assert_eq!(command_env_value(&cmd, "openin_any").as_deref(), Some("p"));
    assert_eq!(command_env_value(&cmd, "openout_any").as_deref(), Some("p"));

    assert!(build_pdflatex_command(Path::new("-shell-escape.tex"), out_dir).is_err());
    assert!(build_pdflatex_command(Path::new("papers/-evil.tex"), out_dir).is_err());
    assert!(build_pdflatex_command(Path::new("papers/--shell-escape.tex"), out_dir).is_err());
    assert!(build_pdflatex_command(tex_path, Path::new("-bad-out")).is_err());
}

#[test]
fn test_build_bibtex_command_hardens_env_and_rejects_dash_stems() {
    let out_dir = Path::new("papers/output");
    let bib_inputs = "/repo/papers:papers//:";
    let cmd = build_bibtex_command("ualbf_quasiperfect", out_dir, bib_inputs)
        .expect("expected valid bibtex command");

    assert_eq!(cmd.get_program(), "bibtex");
    assert_eq!(cmd.get_current_dir(), Some(out_dir));
    assert_eq!(command_args_strings(&cmd), vec!["ualbf_quasiperfect"]);
    assert_eq!(
        command_env_value(&cmd, "BIBINPUTS").as_deref(),
        Some(bib_inputs)
    );
    assert_eq!(
        command_env_value(&cmd, "BSTINPUTS").as_deref(),
        Some(bib_inputs)
    );
    assert_eq!(command_env_value(&cmd, "openin_any").as_deref(), Some("p"));
    assert_eq!(command_env_value(&cmd, "openout_any").as_deref(), Some("p"));

    for bad_stem in [
        "",
        "-",
        "-terse",
        "--version",
        "-min-crossrefs=1",
        "../escape",
    ] {
        assert!(
            build_bibtex_command(bad_stem, out_dir, bib_inputs).is_err(),
            "bibtex builder should reject stem {bad_stem:?}"
        );
    }

    let err = try_run_bibtex_pass("-min-crossrefs=1", out_dir, bib_inputs)
        .expect_err("run_bibtex_pass must reject leading-dash stem before spawning bibtex");
    assert!(err.contains("leading dash"));
}

#[test]
fn test_collect_tex_inputs_rejects_leading_dash_and_accepts_repo_papers() {
    let temp = TempOutputDir::new("tex_inputs");
    let papers_dir = temp.path.join("papers");
    fs::create_dir_all(&papers_dir).unwrap();
    fs::write(papers_dir.join("algebra.tex"), b"\\documentclass{article}").unwrap();
    fs::write(papers_dir.join("references.bib"), b"@article{a,}").unwrap();
    fs::write(
        papers_dir.join("accessibility.sty"),
        b"\\ProvidesPackage{accessibility}",
    )
    .unwrap();

    let valid = collect_tex_inputs(&papers_dir).expect("valid papers dir should pass");
    assert_eq!(valid, vec![papers_dir.join("algebra.tex")]);

    let dash_tex = papers_dir.join("-option.tex");
    fs::write(&dash_tex, b"\\documentclass{article}").unwrap();
    assert!(collect_tex_inputs(&papers_dir).is_err());
    assert!(try_extract_papers_metadata(&[dash_tex]).is_err());

    let repo_papers = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../papers");
    let repo_tex = collect_tex_inputs(&repo_papers).expect("repository papers/ must validate");
    assert!(repo_tex.len() >= 28);
    assert!(repo_tex.iter().any(|p| {
        p.file_name()
            .is_some_and(|n| n == "advanced_linear_algebra.tex")
    }));
}

#[test]
#[cfg(unix)]
fn test_inputs_and_builders_reject_symlinks() {
    use std::os::unix::fs::symlink;

    let temp = TempOutputDir::new("symlinks");
    let outside_secret = temp.path.join("outside.txt");
    fs::write(&outside_secret, b"secret").unwrap();

    // 1. Symlinked papers/ directory is rejected
    let real_dir = temp.path.join("real_papers");
    fs::create_dir_all(&real_dir).unwrap();
    fs::write(real_dir.join("paper.tex"), b"\\documentclass{article}").unwrap();
    let linked_papers_dir = temp.path.join("linked_papers");
    symlink(&real_dir, &linked_papers_dir).unwrap();
    assert!(collect_tex_inputs(&linked_papers_dir).is_err());

    // 2. Symlinked papers/*.tex (valid and dangling) is rejected by collector, metadata, and builder
    let papers_dir = temp.path.join("papers");
    let out_dir = papers_dir.join("output");
    fs::create_dir_all(&out_dir).unwrap();
    fs::write(papers_dir.join("valid.tex"), b"\\documentclass{article}").unwrap();
    let linked_tex = papers_dir.join("evil.tex");
    symlink(&outside_secret, &linked_tex).unwrap();
    assert!(collect_tex_inputs(&papers_dir).is_err());
    assert!(try_extract_papers_metadata(std::slice::from_ref(&linked_tex)).is_err());
    assert!(build_pdflatex_command(&linked_tex, &out_dir).is_err());
    fs::remove_file(&linked_tex).unwrap();

    let dangling_tex = papers_dir.join("dangling.tex");
    symlink(temp.path.join("nonexistent.tex"), &dangling_tex).unwrap();
    assert!(collect_tex_inputs(&papers_dir).is_err());
    fs::remove_file(&dangling_tex).unwrap();

    // 3. Symlinked papers/references.bib (valid and dangling) and output_dir are rejected
    let linked_bib = papers_dir.join("references.bib");
    symlink(&outside_secret, &linked_bib).unwrap();
    assert!(collect_tex_inputs(&papers_dir).is_err());
    fs::remove_file(&linked_bib).unwrap();

    symlink(temp.path.join("missing.bib"), &linked_bib).unwrap();
    assert!(collect_tex_inputs(&papers_dir).is_err());
    fs::remove_file(&linked_bib).unwrap();

    let linked_out = temp.path.join("linked_out");
    symlink(&real_dir, &linked_out).unwrap();
    assert!(build_pdflatex_command(&papers_dir.join("valid.tex"), &linked_out).is_err());
    assert!(build_bibtex_command("valid", &linked_out, "papers//:").is_err());
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

#[test]
fn test_validate_tex_input_allows_nonexistent_synthetic_special_chars() {
    let temp = TempOutputDir::new("synthetic_win32_chars");
    let synthetic = temp.path.join("synthetic_<a>\"b\"?q.tex");
    let stem = validate_tex_input_path(&synthetic)
        .expect("non-existent synthetic path with special chars should not fail symlink check");
    assert_eq!(stem, "synthetic_<a>\"b\"?q");

    let meta = try_extract_papers_metadata(&[synthetic])
        .expect("metadata extraction should succeed for non-existent synthetic path");
    assert_eq!(meta.len(), 1);
    assert_eq!(meta[0].0, "synthetic_<a>\"b\"?q");
}
