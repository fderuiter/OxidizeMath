use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, exit};

/// Compile all academic papers using multi-pass pdflatex and BibTeX,
/// generate an index page for GitHub Pages, and package release archives.
pub fn compile_papers() {
    println!("=== Compiling Academic Papers with pdflatex & BibTeX ===");
    check_prerequisites();

    let papers_dir = Path::new("papers");
    let output_dir = Path::new("papers/output");

    if !output_dir.exists() {
        fs::create_dir_all(output_dir).unwrap();
    }

    let mut tex_files: Vec<PathBuf> = fs::read_dir(papers_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "tex"))
        .collect();
    tex_files.sort();

    println!("Found {} academic papers to compile.", tex_files.len());

    let papers_canonical =
        fs::canonicalize(papers_dir).unwrap_or_else(|_| papers_dir.to_path_buf());
    let bib_inputs = format!("{}:papers//:", papers_canonical.display());

    for path in &tex_files {
        compile_single_paper(path, output_dir, &bib_inputs);
    }

    println!(
        "=== All {} papers compiled successfully ===",
        tex_files.len()
    );

    println!("=== Generating Papers Index Page ===");
    generate_index_html(&tex_files, output_dir);

    println!("=== Packaging Papers Release Archive ===");
    package_release_bundle(output_dir);
}

fn check_prerequisites() {
    let check_pdf = Command::new("pdflatex").arg("--version").output();
    if !check_pdf.map(|o| o.status.success()).unwrap_or(false) {
        eprintln!("Error: pdflatex is not installed or not in PATH.");
        eprintln!(
            "Install TeXLive: sudo apt-get install -y texlive-latex-base texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra texlive-science"
        );
        exit(1);
    }

    let check_bib = Command::new("bibtex").arg("--version").output();
    if !check_bib.map(|o| o.status.success()).unwrap_or(false) {
        eprintln!("Error: bibtex is not installed or not in PATH.");
        eprintln!("Install TeXLive binaries: sudo apt-get install -y texlive-binaries");
        exit(1);
    }
}

fn compile_single_paper(path: &Path, output_dir: &Path, bib_inputs: &str) {
    let stem = path.file_stem().unwrap().to_string_lossy();
    println!("Compiling {:?}...", path);

    // Pass 1: Initial pdflatex run to generate aux file
    run_pdflatex(path, output_dir);

    // BibTeX pass if citations exist
    let aux_path = output_dir.join(format!("{}.aux", stem));
    if aux_path.exists() {
        let aux_content = fs::read_to_string(&aux_path).unwrap_or_default();
        if aux_content.contains("\\citation") && aux_content.contains("\\bibdata") {
            run_bibtex_pass(&stem, output_dir, bib_inputs);
            // Pass 2: Re-run pdflatex to incorporate bibliography
            run_pdflatex(path, output_dir);
        }
    }

    // Pass 3+: Re-run pdflatex until cross-references stabilize
    let mut pass = 2;
    loop {
        pass += 1;
        run_pdflatex(path, output_dir);

        let log_path = output_dir.join(format!("{}.log", stem));
        let log = fs::read_to_string(&log_path).unwrap_or_default();

        let needs_rerun = log.contains("Rerun to get cross-references right")
            || log.contains("Label(s) may have changed");

        if !needs_rerun || pass >= 5 {
            break;
        }
    }

    let log_path = output_dir.join(format!("{}.log", stem));
    verify_paper_log(path, &log_path);
    println!("  Successfully compiled and verified {:?}", path);
}

fn run_bibtex_pass(stem: &str, output_dir: &Path, bib_inputs: &str) {
    println!("  Running BibTeX for {}...", stem);
    let bib_status = Command::new("bibtex")
        .current_dir(output_dir)
        .env("BIBINPUTS", bib_inputs)
        .env("BSTINPUTS", bib_inputs)
        .arg(stem)
        .output();

    match bib_status {
        Ok(out) if out.status.success() => {
            let blg_path = output_dir.join(format!("{}.blg", stem));
            if blg_path.exists() {
                let blg = fs::read_to_string(&blg_path).unwrap_or_default();
                if blg.contains("Warning--I didn't find a database entry")
                    || blg.contains("error message")
                {
                    eprintln!("Error: BibTeX unresolved references in {}:\n{}", stem, blg);
                    exit(1);
                }
            }
        }
        Ok(out) => {
            eprintln!(
                "Error: BibTeX failed for {}:\n{}",
                stem,
                String::from_utf8_lossy(&out.stdout)
            );
            exit(1);
        }
        Err(e) => {
            eprintln!("Failed to execute bibtex for {}: {}", stem, e);
            exit(1);
        }
    }
}

fn verify_paper_log(path: &Path, log_path: &Path) {
    let log = fs::read_to_string(log_path).unwrap_or_default();
    for line in log.lines() {
        if (line.contains("Citation") || line.contains("Reference")) && line.contains("undefined") {
            eprintln!(
                "Error: Unresolved citation/reference in {:?}: {}",
                path, line
            );
            exit(1);
        }
        if line.contains("There were undefined references.") {
            eprintln!("Error: Undefined references found in {:?}", path);
            exit(1);
        }
    }
}

fn run_pdflatex(path: &Path, output_dir: &Path) {
    let status = Command::new("pdflatex")
        .env("TEXINPUTS", "papers//:")
        .arg("-interaction=nonstopmode")
        .arg("-halt-on-error")
        .arg("-output-directory")
        .arg(output_dir)
        .arg(path)
        .status();

    match status {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("Failed to compile {:?}", path);
            exit(1);
        }
    }
}

fn is_safe_bundle_filename(name: &str) -> bool {
    !name.starts_with('-')
        && (name.ends_with(".pdf")
            || name == "Verification_Certificate.json"
            || name == "index.html")
        && !name.starts_with("SHA256SUMS")
}

fn collect_bundle_artifacts(output_dir: &Path) -> Vec<String> {
    let entries = fs::read_dir(output_dir).unwrap();
    let mut artifacts = Vec::new();
    for entry in entries.filter_map(|e| e.ok()) {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('-') {
            eprintln!("Warning: ignoring filename with leading dash: {}", name);
            continue;
        }
        if is_safe_bundle_filename(&name) {
            artifacts.push(name);
        }
    }
    artifacts.sort();
    artifacts.dedup();
    artifacts
}

fn build_tar_args(files_to_pack: &[String]) -> Vec<&str> {
    let mut tar_args = vec!["-czf", "papers-bundle.tar.gz", "--"];
    let file_refs: Vec<&str> = files_to_pack.iter().map(String::as_str).collect();
    tar_args.extend(file_refs);
    tar_args
}

fn build_checksum_args<'a>(cmd: &str, checksum_targets: &'a [String]) -> Vec<&'a str> {
    let mut cmd_args: Vec<&'a str> = Vec::new();
    if cmd == "shasum" {
        cmd_args.push("-a");
        cmd_args.push("256");
    }
    cmd_args.push("--");
    let target_refs: Vec<&'a str> = checksum_targets.iter().map(String::as_str).collect();
    cmd_args.extend(target_refs);
    cmd_args
}

fn detect_sha256_cmd() -> &'static str {
    if Command::new("sha256sum").arg("--version").output().is_ok() {
        "sha256sum"
    } else {
        "shasum"
    }
}

fn build_check_args<'a>(cmd: &str, manifest_name: &'a str) -> Vec<&'a str> {
    if cmd == "shasum" {
        vec!["-a", "256", "-c", "--", manifest_name]
    } else {
        vec!["-c", "--", manifest_name]
    }
}

pub fn package_release_bundle(output_dir: &Path) {
    let cert_path = Path::new("Verification_Certificate.json");
    if cert_path.exists() {
        fs::copy(cert_path, output_dir.join("Verification_Certificate.json"))
            .expect("Failed to copy Verification_Certificate.json");
    }

    let inner_artifacts = collect_bundle_artifacts(output_dir);
    write_sha256_manifest(output_dir, "SHA256SUMS", &inner_artifacts);

    let mut files_to_pack = inner_artifacts;
    files_to_pack.push("SHA256SUMS".to_string());
    files_to_pack.sort();

    let bundle_file = output_dir.join("papers-bundle.tar.gz");
    let tar_args = build_tar_args(&files_to_pack);
    let tar_status = Command::new("tar")
        .current_dir(output_dir)
        .args(&tar_args)
        .status();

    if !tar_status.map(|s| s.success()).unwrap_or(false) {
        eprintln!("Error creating papers bundle archive.");
        exit(1);
    }
    println!("Created release bundle at {:?}", bundle_file);

    let outer_targets = vec!["papers-bundle.tar.gz".to_string()];
    write_sha256_manifest(output_dir, "SHA256SUMS-papers", &outer_targets);
}

fn write_sha256_manifest(output_dir: &Path, manifest_name: &str, checksum_targets: &[String]) {
    if checksum_targets.is_empty() {
        eprintln!("Error: no checksum targets found for {}.", manifest_name);
        exit(1);
    }

    let cmd = detect_sha256_cmd();
    let cmd_args = build_checksum_args(cmd, checksum_targets);
    let output = Command::new(cmd)
        .current_dir(output_dir)
        .args(&cmd_args)
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let sums_path = output_dir.join(manifest_name);
            fs::write(&sums_path, &out.stdout).unwrap();
            println!("Generated {} at {:?}", manifest_name, sums_path);

            let check_args = build_check_args(cmd, manifest_name);
            let check_status = Command::new(cmd)
                .current_dir(output_dir)
                .args(&check_args)
                .status();
            if !check_status.map(|s| s.success()).unwrap_or(false) {
                eprintln!("Checksum verification failed for {}!", manifest_name);
                exit(1);
            }
            println!("{} successfully verified.", manifest_name);
        }
        _ => {
            eprintln!("Error generating SHA-256 sums for {}.", manifest_name);
            exit(1);
        }
    }
}

pub fn generate_index_html(tex_files: &[PathBuf], output_dir: &Path) {
    let mut papers_meta = extract_papers_metadata(tex_files);
    papers_meta.sort_by(|a, b| a.1.cmp(&b.1));

    let mut html = String::from(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OxidizeMath Academic Papers</title>
  <style>
    :root { --bg: #0d1117; --card-bg: #161b22; --card-border: #30363d; --text: #c9d1d9; --heading: #f0f6fc; --accent: #58a6ff; --accent-hover: #79c0ff; --meta: #8b949e; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; background-color: var(--bg); color: var(--text); line-height: 1.6; margin: 0; padding: 2rem 1rem; }
    .container { max-width: 1100px; margin: 0 auto; }
    header { margin-bottom: 2.5rem; border-bottom: 1px solid var(--card-border); padding-bottom: 1.5rem; }
    h1 { color: var(--heading); font-size: 2.25rem; margin: 0 0 0.5rem 0; }
    .subtitle { color: var(--meta); font-size: 1.1rem; margin: 0 0 1rem 0; }
    .actions-bar { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1rem; }
    .btn { display: inline-block; padding: 0.5rem 1rem; background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: 6px; color: var(--accent); text-decoration: none; font-weight: 500; transition: all 0.2s ease; }
    .btn:hover { background-color: #21262d; border-color: var(--accent); color: var(--accent-hover); }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1.5rem; }
    .card { background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 1.25rem; display: flex; flex-direction: column; justify-content: space-between; }
    .card-title { color: var(--heading); font-size: 1.2rem; margin: 0 0 0.5rem 0; }
    .card-meta { font-size: 0.85rem; color: var(--meta); margin-bottom: 0.75rem; }
    .card-abstract { font-size: 0.9rem; color: var(--text); margin-bottom: 1.25rem; flex-grow: 1; }
    .card-links { display: flex; justify-content: space-between; align-items: center; margin-top: auto; }
    footer { margin-top: 4rem; text-align: center; color: var(--meta); font-size: 0.85rem; border-top: 1px solid var(--card-border); padding-top: 2rem; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>OxidizeMath Academic Papers</h1>
      <p class="subtitle">Permanent theoretical foundation and empirical verification publications (28 papers)</p>
      <div class="actions-bar">
        <a class="btn" href="papers-bundle.tar.gz">📦 Download All Papers (.tar.gz)</a>
        <a class="btn" href="SHA256SUMS">🔒 Checksum Manifest (SHA256)</a>
        <a class="btn" href="https://github.com/fderuiter/OxidizeMath">💻 GitHub Repository</a>
      </div>
    </header>
    <main class="grid">
"#,
    );

    for (stem, title, author, abstract_text) in papers_meta {
        html.push_str(&render_paper_card(&stem, &title, &author, &abstract_text));
    }

    html.push_str(
        r#"    </main>
    <footer>
      <p>Published by OxidizeMath Automated Continuous Verification &amp; Publishing System</p>
    </footer>
  </div>
</body>
</html>
"#,
    );

    let index_path = output_dir.join("index.html");
    fs::write(&index_path, html).unwrap();
    println!("Generated papers index page at {:?}", index_path);
}

fn extract_papers_metadata(tex_files: &[PathBuf]) -> Vec<(String, String, String, String)> {
    let mut papers_meta = Vec::new();
    for path in tex_files {
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let content = fs::read_to_string(path).unwrap_or_default();

        let title = extract_latex_command(&content, "\\title")
            .unwrap_or_else(|| format_title_from_stem(&stem));
        let author = extract_latex_command(&content, "\\author")
            .unwrap_or_else(|| "OxidizeMath Project".to_string());
        let abstract_text = extract_latex_environment(&content, "abstract")
            .unwrap_or_else(|| "Mathematical formulation and empirical verification.".to_string());

        papers_meta.push((stem, title, author, abstract_text));
    }
    papers_meta
}

fn render_paper_card(stem: &str, title: &str, author: &str, abstract_text: &str) -> String {
    let escaped_title = escape_html(title);
    let escaped_author = escape_html(author);
    let escaped_abstract = escape_html(abstract_text);

    format!(
        r#"      <article class="card">
        <div>
          <h2 class="card-title">{}</h2>
          <div class="card-meta">By {} &bull; <code>{}.pdf</code></div>
          <p class="card-abstract">{}</p>
        </div>
        <div class="card-links">
          <a class="btn" href="{}.pdf" target="_blank" rel="noopener">📄 Read PDF</a>
        </div>
      </article>
"#,
        escaped_title, escaped_author, stem, escaped_abstract, stem
    )
}

fn extract_latex_command(content: &str, cmd: &str) -> Option<String> {
    let idx = content.find(cmd)?;
    let after_cmd = &content[idx + cmd.len()..];
    let open_brace = after_cmd.find('{')?;
    let start = &after_cmd[open_brace + 1..];

    let mut depth = 1;
    let mut end = 0;
    for (i, c) in start.char_indices() {
        if c == '{' {
            depth += 1;
        } else if c == '}' {
            depth -= 1;
            if depth == 0 {
                end = i;
                break;
            }
        }
    }

    if depth == 0 {
        Some(clean_latex(&start[..end]))
    } else {
        None
    }
}

fn extract_latex_environment(content: &str, env: &str) -> Option<String> {
    let begin_tag = format!("\\begin{{{}}}", env);
    let end_tag = format!("\\end{{{}}}", env);

    let start_idx = content.find(&begin_tag)?;
    let after_begin = &content[start_idx + begin_tag.len()..];
    let end_idx = after_begin.find(&end_tag)?;

    Some(clean_latex(&after_begin[..end_idx]))
}

fn clean_latex(s: &str) -> String {
    let s = s
        .replace("\\texttt{", "")
        .replace("\\textbf{", "")
        .replace("\\textit{", "");
    let s = s.replace(['}', '{'], "");
    let s = s.replace("\\\\", " ").replace('\n', " ");
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn format_title_from_stem(stem: &str) -> String {
    if stem.eq_ignore_ascii_case("grpo") {
        return "Group Relative Policy Optimization (GRPO)".to_string();
    }
    stem.replace('_', " ")
        .split_whitespace()
        .map(|word| {
            let mut c = word.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

#[cfg(test)]
mod tests;
