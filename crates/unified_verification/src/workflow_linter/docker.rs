use serde_json::{Map, Value};
use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};
use walkdir::WalkDir;

fn is_sixty_four_char_hex(s: &str) -> bool {
    s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit())
}

pub(super) fn is_pinned_docker_uri(uri: &str) -> bool {
    let Some(rest) = uri.trim().strip_prefix("docker://") else {
        return false;
    };
    let Some((image, digest)) = rest.rsplit_once("@sha256:") else {
        return false;
    };
    !image.is_empty()
        && !image.starts_with('/')
        && !image.starts_with(':')
        && !image.contains("..")
        && !image.chars().any(char::is_whitespace)
        && is_sixty_four_char_hex(digest)
}

pub(super) fn extract_safe_local_rel_path(
    local_ref: &str,
    allow_trailing_slash: bool,
) -> Result<PathBuf, String> {
    let trimmed = local_ref.trim();
    if !trimmed.starts_with("./") {
        return Err(format!(
            "local reference '{}' must start with './'.",
            local_ref
        ));
    }
    if trimmed.contains('\\') || trimmed.contains('\0') {
        return Err(format!(
            "local reference '{}' contains invalid path characters.",
            local_ref
        ));
    }
    if trimmed.contains('@') {
        return Err(format!(
            "local reference '{}' must not include an '@' version ref.",
            local_ref
        ));
    }
    let raw_rel = &trimmed[2..];
    if !allow_trailing_slash && raw_rel.ends_with('/') {
        return Err(format!(
            "local file reference '{}' must not end with '/'.",
            local_ref
        ));
    }
    let stripped = if allow_trailing_slash {
        raw_rel.trim_end_matches('/')
    } else {
        raw_rel
    };
    if stripped.is_empty() || stripped.split('/').any(|seg| seg == "..") {
        return Err(format!(
            "local reference '{}' must not contain '..' or empty path segments.",
            local_ref
        ));
    }
    if stripped.split('/').any(|seg| seg.is_empty() || seg == ".") {
        return Err(format!(
            "local reference '{}' contains redundant or empty path segments.",
            local_ref
        ));
    }
    let mut normalized = PathBuf::new();
    for component in Path::new(stripped).components() {
        match component {
            Component::Normal(seg) => normalized.push(seg),
            _ => {
                return Err(format!(
                    "local reference '{}' must stay within the repository root without '..' segments.",
                    local_ref
                ));
            }
        }
    }
    Ok(normalized)
}

pub(super) fn canonicalize_within(
    target: &Path,
    boundary: &Path,
    raw_ref: &str,
    boundary_desc: &str,
) -> Result<PathBuf, String> {
    let canonical_boundary = fs::canonicalize(boundary).map_err(|_| {
        format!(
            "{} '{}' does not exist on disk while resolving '{}'.",
            boundary_desc,
            boundary.display(),
            raw_ref
        )
    })?;
    let canonical_target = fs::canonicalize(target)
        .map_err(|_| format!("referenced path '{}' does not exist on disk.", raw_ref))?;
    if !canonical_target.starts_with(&canonical_boundary) {
        return Err(format!(
            "reference '{}' escapes {} ('{}').",
            raw_ref,
            boundary_desc,
            boundary.display()
        ));
    }
    Ok(canonical_target)
}

fn validate_from_instruction(
    rest: &str,
    stages: &mut HashSet<String>,
    dockerfile_path: &Path,
    manifest_label: &str,
) -> Result<(), String> {
    let tokens: Vec<&str> = rest
        .split_whitespace()
        .filter(|tok| !tok.starts_with("--"))
        .collect();
    let Some(&image_spec) = tokens.first() else {
        return Err(format!(
            "Malformed FROM instruction in Dockerfile '{}' for '{}'.",
            dockerfile_path.display(),
            manifest_label
        ));
    };
    let pinned = image_spec.eq_ignore_ascii_case("scratch")
        || stages.contains(&image_spec.to_ascii_lowercase())
        || image_spec
            .rsplit_once("@sha256:")
            .is_some_and(|(img, dig)| !img.is_empty() && is_sixty_four_char_hex(dig));

    if !pinned {
        return Err(format!(
            "Dockerfile '{}' in '{}' uses unpinned base image '{}'; pin base images to '@sha256:<64-hex>'.",
            dockerfile_path.display(),
            manifest_label,
            image_spec
        ));
    }
    if tokens.len() >= 3 && tokens[1].eq_ignore_ascii_case("as") {
        stages.insert(tokens[2].to_ascii_lowercase());
    }
    Ok(())
}

fn validate_dockerfile_pinning(dockerfile_path: &Path, manifest_label: &str) -> Result<(), String> {
    let content = fs::read_to_string(dockerfile_path).map_err(|e| {
        format!(
            "Failed to read Dockerfile '{}' for '{}': {}",
            dockerfile_path.display(),
            manifest_label,
            e
        )
    })?;
    let mut stages = HashSet::new();
    let mut saw_from = false;

    for raw_line in content.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.len() > 5 && trimmed[..5].eq_ignore_ascii_case("from ") {
            saw_from = true;
            validate_from_instruction(&trimmed[5..], &mut stages, dockerfile_path, manifest_label)?;
        }
    }
    if !saw_from {
        return Err(format!(
            "Dockerfile '{}' in '{}' contains no FROM instruction.",
            dockerfile_path.display(),
            manifest_label
        ));
    }
    Ok(())
}

pub(super) fn validate_action_runs_image(
    runs_map: &Map<String, Value>,
    action_dir: &Path,
    repo_root: &Path,
    manifest_label: &str,
) -> Result<(), String> {
    let is_docker = runs_map.get("using").and_then(Value::as_str) == Some("docker");
    let image_val = runs_map.get("image");
    if is_docker && image_val.is_none() {
        return Err(format!(
            "Docker action manifest '{}' is missing 'runs.image'.",
            manifest_label
        ));
    }
    let Some(val) = image_val else {
        return Ok(());
    };
    let Some(image_str) = val.as_str() else {
        return Err(format!(
            "Action manifest '{}' has a non-string 'runs.image' attribute.",
            manifest_label
        ));
    };
    let trimmed = image_str.trim();
    if trimmed.is_empty() {
        return Err(format!(
            "Action manifest '{}' has an empty 'runs.image' attribute.",
            manifest_label
        ));
    }
    if trimmed.starts_with("docker://") {
        if !is_pinned_docker_uri(trimmed) {
            return Err(format!(
                "Action manifest '{}' has unpinned Docker image '{}'; pin to 'docker://<image>@sha256:<64-hex>'.",
                manifest_label, image_str
            ));
        }
        return Ok(());
    }
    let normalized_ref = if trimmed.starts_with("./") {
        trimmed.to_string()
    } else {
        format!("./{}", trimmed)
    };
    let rel_df = extract_safe_local_rel_path(&normalized_ref, false).map_err(|_| {
        format!(
            "Action manifest '{}' has invalid or unpinned 'runs.image' '{}'; use 'docker://<image>@sha256:<64-hex>'.",
            manifest_label, image_str
        )
    })?;
    let df_path = action_dir.join(rel_df);
    if !df_path.is_file() {
        return Err(format!(
            "Action manifest '{}' has unpinned or missing 'runs.image' '{}'; pin external images to 'docker://<image>@sha256:<64-hex>'.",
            manifest_label, image_str
        ));
    }
    canonicalize_within(&df_path, repo_root, image_str, "repository root")?;
    let canonical_df = canonicalize_within(&df_path, action_dir, image_str, "action directory")?;
    validate_dockerfile_pinning(&canonical_df, manifest_label)
}

pub(super) fn infer_repo_and_actions_dirs(workflows_dir: &Path) -> (PathBuf, PathBuf) {
    if workflows_dir.file_name().and_then(|n| n.to_str()) == Some("workflows")
        && let Some(dot_github) = workflows_dir.parent()
        && dot_github.file_name().and_then(|n| n.to_str()) == Some(".github")
    {
        let repo_root = match dot_github.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        };
        return (repo_root, dot_github.join("actions"));
    }
    (
        workflows_dir.to_path_buf(),
        workflows_dir.join(".github").join("actions"),
    )
}

pub(super) fn lint_actions_dir(actions_dir: &Path, repo_root: &Path) -> bool {
    if !actions_dir.exists() {
        return true;
    }
    let mut all_passed = true;
    let mut state = super::pinning::PinningState::default();

    for entry in WalkDir::new(actions_dir).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        let path_str = path.to_string_lossy();
        if entry.file_type().is_symlink()
            && let Err(msg) = canonicalize_within(path, repo_root, &path_str, "repository root")
        {
            eprintln!("Error in {}: {}", path_str, msg);
            all_passed = false;
            continue;
        }
        let is_manifest = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n == "action.yml" || n == "action.yaml");
        if path.is_file()
            && is_manifest
            && let Err(msg) =
                super::pinning::validate_action_manifest_with_state(path, repo_root, &mut state)
        {
            eprintln!("Error in {}: {}", path_str, msg);
            all_passed = false;
        }
    }

    all_passed
}
