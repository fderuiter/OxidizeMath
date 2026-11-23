mod bundle;
mod index_html;

pub use bundle::package_release_bundle;
#[cfg(test)]
use bundle::{
    build_checksum_args, build_tar_args, collect_bundle_artifacts, is_safe_bundle_filename,
};
use index_html::{
    INDEX_HTML_FOOTER, IndexTarget, extract_latex_command, extract_latex_environment,
    format_title_from_stem, render_index_header, render_paper_card_for_target,
};
#[cfg(test)]
use index_html::{encode_uri_component, render_paper_card};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::{Command, exit};

/// Compile all academic papers using multi-pass pdflatex and BibTeX,
/// generate an index page for GitHub Pages, and package release archives.
pub fn compile_papers() {
    println!("=== Compiling Academic Papers with pdflatex & BibTeX ===");
    check_prerequisites();

    let papers_dir = Path::new("papers");
    let output_dir = Path::new("papers/output");

    let tex_files = collect_tex_inputs(papers_dir).unwrap_or_else(|err| {
        eprintln!("Error: {err}");
        exit(1);
    });

    if let Err(err) = ensure_safe_output_dir(output_dir) {
        eprintln!("Error: {err}");
        exit(1);
    }

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

    println!("=== Generating Papers Index Pages ===");
    generate_index_html(&tex_files, output_dir);

    println!("=== Packaging Papers Release Archive ===");
    package_release_bundle(output_dir);
}

fn reject_if_symlink(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => {
            Err(format!("Refusing symbolic link path: {}", path.display()))
        }
        Ok(_) => Ok(()),
        Err(err) if matches!(err.kind(), ErrorKind::NotFound | ErrorKind::InvalidFilename) => {
            Ok(())
        }
        Err(err) => Err(format!(
            "Failed to inspect metadata for {}: {}",
            path.display(),
            err
        )),
    }
}

fn validate_output_dir_path(output_dir: &Path) -> Result<(), String> {
    if output_dir.as_os_str().is_empty() {
        return Err("Output directory path must not be empty".to_string());
    }
    if output_dir.as_os_str().to_string_lossy().starts_with('-') {
        return Err(format!(
            "Refusing output directory with leading dash: {}",
            output_dir.display()
        ));
    }
    reject_if_symlink(output_dir)
}

fn ensure_safe_output_dir(output_dir: &Path) -> Result<(), String> {
    validate_output_dir_path(output_dir)?;
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).map_err(|err| {
            format!(
                "Failed to create output directory {}: {}",
                output_dir.display(),
                err
            )
        })?;
    }
    Ok(())
}

fn validate_paper_stem(stem: &str) -> Result<(), String> {
    if stem.is_empty() {
        return Err("Paper stem must not be empty".to_string());
    }
    if stem.starts_with('-') {
        return Err(format!("Refusing paper stem with leading dash: {stem}"));
    }
    if stem.contains('/') || stem.contains('\\') || stem == "." || stem == ".." {
        return Err(format!(
            "Refusing paper stem with path traversal components: {stem}"
        ));
    }
    Ok(())
}

fn validate_tex_input_path(path: &Path) -> Result<String, String> {
    let raw_path = path.as_os_str().to_string_lossy();
    if raw_path.is_empty() {
        return Err("TeX input path must not be empty".to_string());
    }
    if raw_path.starts_with('-') {
        return Err(format!(
            "Refusing TeX input path with leading dash: {}",
            path.display()
        ));
    }
    reject_if_symlink(path)?;

    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy())
        .ok_or_else(|| format!("Missing file name in TeX path: {}", path.display()))?;
    if file_name.starts_with('-') {
        return Err(format!(
            "Refusing TeX file name with leading dash: {}",
            path.display()
        ));
    }
    if !path.extension().is_some_and(|ext| ext == "tex") {
        return Err(format!(
            "Expected .tex extension for TeX input: {}",
            path.display()
        ));
    }

    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .ok_or_else(|| format!("Missing file stem in TeX path: {}", path.display()))?;
    validate_paper_stem(&stem)?;
    Ok(stem)
}

fn validate_papers_dir_and_bib(papers_dir: &Path) -> Result<(), String> {
    let dir_meta = fs::symlink_metadata(papers_dir).map_err(|err| {
        format!(
            "Failed to inspect papers directory {}: {}",
            papers_dir.display(),
            err
        )
    })?;
    if dir_meta.file_type().is_symlink() {
        return Err(format!(
            "Refusing symbolic link for papers directory: {}",
            papers_dir.display()
        ));
    }
    if !dir_meta.is_dir() {
        return Err(format!(
            "Papers path is not a directory: {}",
            papers_dir.display()
        ));
    }

    let bib_path = papers_dir.join("references.bib");
    match fs::symlink_metadata(&bib_path) {
        Ok(meta) if meta.file_type().is_symlink() => Err(format!(
            "Refusing symbolic link for bibliography file: {}",
            bib_path.display()
        )),
        Ok(meta) if !meta.is_file() => Err(format!(
            "Bibliography path is not a regular file: {}",
            bib_path.display()
        )),
        Ok(_) => Ok(()),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
        Err(err) => Err(format!(
            "Failed to inspect bibliography file {}: {}",
            bib_path.display(),
            err
        )),
    }
}

fn collect_tex_inputs(papers_dir: &Path) -> Result<Vec<PathBuf>, String> {
    validate_papers_dir_and_bib(papers_dir)?;

    let entries = fs::read_dir(papers_dir).map_err(|err| {
        format!(
            "Failed to read papers directory {}: {}",
            papers_dir.display(),
            err
        )
    })?;

    let mut tex_files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| {
            format!(
                "Failed to read directory entry in {}: {}",
                papers_dir.display(),
                err
            )
        })?;
        let path = entry.path();
        let meta = fs::symlink_metadata(&path)
            .map_err(|err| format!("Failed to inspect metadata for {}: {}", path.display(), err))?;
        if meta.file_type().is_symlink() {
            return Err(format!(
                "Refusing symbolic link in papers directory: {}",
                path.display()
            ));
        }
        if path.extension().is_some_and(|ext| ext == "tex") {
            if !meta.is_file() {
                return Err(format!(
                    "TeX input is not a regular file: {}",
                    path.display()
                ));
            }
            validate_tex_input_path(&path)?;
            tex_files.push(path);
        }
    }

    tex_files.sort();
    Ok(tex_files)
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
    let stem = validate_tex_input_path(path).unwrap_or_else(|err| {
        eprintln!("Error: {err}");
        exit(1);
    });
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

fn build_bibtex_command(
    stem: &str,
    output_dir: &Path,
    bib_inputs: &str,
) -> Result<Command, String> {
    validate_paper_stem(stem)?;
    validate_output_dir_path(output_dir)?;

    let mut cmd = Command::new("bibtex");
    cmd.current_dir(output_dir)
        .env("BIBINPUTS", bib_inputs)
        .env("BSTINPUTS", bib_inputs)
        .env("openin_any", "p")
        .env("openout_any", "p")
        .arg(stem);
    Ok(cmd)
}

fn try_run_bibtex_pass(stem: &str, output_dir: &Path, bib_inputs: &str) -> Result<(), String> {
    let mut cmd = build_bibtex_command(stem, output_dir, bib_inputs)?;
    println!("  Running BibTeX for {}...", stem);
    let out = cmd
        .output()
        .map_err(|e| format!("Failed to execute bibtex for {stem}: {e}"))?;

    if !out.status.success() {
        return Err(format!(
            "Error: BibTeX failed for {}:\n{}",
            stem,
            String::from_utf8_lossy(&out.stdout)
        ));
    }

    let blg_path = output_dir.join(format!("{}.blg", stem));
    if blg_path.exists() {
        let blg = fs::read_to_string(&blg_path).unwrap_or_default();
        if blg.contains("Warning--I didn't find a database entry") || blg.contains("error message")
        {
            return Err(format!(
                "Error: BibTeX unresolved references in {stem}:\n{blg}"
            ));
        }
    }
    Ok(())
}

fn run_bibtex_pass(stem: &str, output_dir: &Path, bib_inputs: &str) {
    if let Err(err) = try_run_bibtex_pass(stem, output_dir, bib_inputs) {
        eprintln!("{err}");
        exit(1);
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

fn build_pdflatex_command(path: &Path, output_dir: &Path) -> Result<Command, String> {
    validate_tex_input_path(path)?;
    validate_output_dir_path(output_dir)?;

    let mut cmd = Command::new("pdflatex");
    cmd.env("TEXINPUTS", "papers//:")
        .env("openin_any", "p")
        .env("openout_any", "p")
        .arg("-no-shell-escape")
        .arg("-interaction=nonstopmode")
        .arg("-halt-on-error")
        .arg("-output-directory")
        .arg(output_dir)
        .arg(path);
    Ok(cmd)
}

fn try_run_pdflatex(path: &Path, output_dir: &Path) -> Result<(), String> {
    let mut cmd = build_pdflatex_command(path, output_dir)?;
    match cmd.status() {
        Ok(s) if s.success() => Ok(()),
        _ => Err(format!("Failed to compile {:?}", path)),
    }
}

fn run_pdflatex(path: &Path, output_dir: &Path) {
    if let Err(err) = try_run_pdflatex(path, output_dir) {
        eprintln!("{err}");
        exit(1);
    }
}

fn render_index_document(
    papers_meta: &[(String, String, String, String)],
    target: IndexTarget,
) -> String {
    let mut html = render_index_header(target);
    for (stem, title, author, abstract_text) in papers_meta {
        html.push_str(&render_paper_card_for_target(
            stem,
            title,
            author,
            abstract_text,
            target,
        ));
    }
    html.push_str(INDEX_HTML_FOOTER);
    html
}

/// Generate static HTML index pages for the release archive and GitHub Pages.
pub fn generate_index_html(tex_files: &[PathBuf], output_dir: &Path) {
    let mut papers_meta = extract_papers_metadata(tex_files);
    papers_meta.sort_by(|a, b| a.1.cmp(&b.1));

    let archive_path = output_dir.join("index.html");
    let archive_html = render_index_document(&papers_meta, IndexTarget::Archive);
    fs::write(&archive_path, archive_html).unwrap();
    println!("Generated archive papers index page at {:?}", archive_path);

    let pages_path = output_dir.join("pages-index.html");
    let pages_html = render_index_document(&papers_meta, IndexTarget::GitHubPages);
    fs::write(&pages_path, pages_html).unwrap();
    println!(
        "Generated GitHub Pages papers index page at {:?}",
        pages_path
    );
}

fn try_extract_papers_metadata(
    tex_files: &[PathBuf],
) -> Result<Vec<(String, String, String, String)>, String> {
    let mut papers_meta = Vec::with_capacity(tex_files.len());
    for path in tex_files {
        let stem = validate_tex_input_path(path)?;
        let content = fs::read_to_string(path).unwrap_or_default();

        let title = extract_latex_command(&content, "\\title")
            .unwrap_or_else(|| format_title_from_stem(&stem));
        let author = extract_latex_command(&content, "\\author")
            .unwrap_or_else(|| "OxidizeMath Project".to_string());
        let abstract_text = extract_latex_environment(&content, "abstract")
            .unwrap_or_else(|| "Mathematical formulation and empirical verification.".to_string());

        papers_meta.push((stem, title, author, abstract_text));
    }
    Ok(papers_meta)
}

fn extract_papers_metadata(tex_files: &[PathBuf]) -> Vec<(String, String, String, String)> {
    try_extract_papers_metadata(tex_files).unwrap_or_else(|err| {
        eprintln!("Error: {err}");
        exit(1);
    })
}

#[cfg(test)]
mod tests;
