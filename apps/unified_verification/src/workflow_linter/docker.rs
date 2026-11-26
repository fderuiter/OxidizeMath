use serde_json::{Map, Value};
use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};
use walkdir::WalkDir;

fn is_sixty_four_char_hex(s: &str) -> bool {
    s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit())
}

fn is_pinned_image_digest(spec: &str) -> bool {
    let Some((image, digest)) = spec.rsplit_once("@sha256:") else {
        return false;
    };
    !image.is_empty()
        && !image.starts_with('/')
        && !image.starts_with(':')
        && !image.contains("..")
        && !image.chars().any(char::is_whitespace)
        && is_sixty_four_char_hex(digest)
}

pub(super) fn is_pinned_docker_uri(uri: &str) -> bool {
    uri.trim()
        .strip_prefix("docker://")
        .is_some_and(is_pinned_image_digest)
}

pub(super) fn extract_safe_local_rel_path(
    local_ref: &str,
    allow_trailing_slash: bool,
) -> Result<PathBuf, String> {
    let trimmed = local_ref.trim();
    if !trimmed.starts_with("./") {
        return Err(format!(
            "local reference '{local_ref}' must start with './'."
        ));
    }
    if trimmed.contains('\\') || trimmed.contains('\0') {
        return Err(format!(
            "local reference '{local_ref}' contains invalid path characters."
        ));
    }
    if trimmed.contains('@') {
        return Err(format!(
            "local reference '{local_ref}' must not include an '@' version ref."
        ));
    }
    let raw_rel = &trimmed[2..];
    if !allow_trailing_slash && raw_rel.ends_with('/') {
        return Err(format!(
            "local file reference '{local_ref}' must not end with '/'."
        ));
    }
    let stripped = if allow_trailing_slash {
        raw_rel.trim_end_matches('/')
    } else {
        raw_rel
    };
    if stripped.is_empty() || stripped.split('/').any(|seg| seg == "..") {
        return Err(format!(
            "local reference '{local_ref}' must not contain '..' or empty path segments."
        ));
    }
    if stripped.split('/').any(|seg| seg.is_empty() || seg == ".") {
        return Err(format!(
            "local reference '{local_ref}' contains redundant or empty path segments."
        ));
    }
    let mut normalized = PathBuf::new();
    for component in Path::new(stripped).components() {
        match component {
            Component::Normal(seg) => normalized.push(seg),
            _ => {
                return Err(format!(
                    "local reference '{local_ref}' must stay within the repository root without '..' segments."
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
    let b_disp = boundary.display();
    let canonical_boundary = fs::canonicalize(boundary).map_err(|_| {
        format!("{boundary_desc} '{b_disp}' does not exist on disk while resolving '{raw_ref}'.")
    })?;
    let canonical_target = fs::canonicalize(target)
        .map_err(|_| format!("referenced path '{raw_ref}' does not exist on disk."))?;
    if !canonical_target.starts_with(&canonical_boundary) {
        return Err(format!(
            "reference '{raw_ref}' escapes {boundary_desc} ('{b_disp}')."
        ));
    }
    Ok(canonical_target)
}

#[derive(Default)]
struct DockerfileStages {
    count: usize,
    aliases: HashSet<String>,
}

fn is_valid_stage_index(spec: &str, stage_count: usize) -> bool {
    !spec.is_empty()
        && spec.chars().all(|c| c.is_ascii_digit())
        && spec.parse::<usize>().is_ok_and(|idx| idx < stage_count)
}

fn is_valid_image_or_stage(spec: &str, stages: &DockerfileStages, allow_index: bool) -> bool {
    if spec.is_empty() {
        return false;
    }
    let is_local = !spec.contains('$')
        && (spec.eq_ignore_ascii_case("scratch")
            || stages.aliases.contains(&spec.to_ascii_lowercase())
            || (allow_index && is_valid_stage_index(spec, stages.count)));
    is_local || is_pinned_image_digest(spec)
}

fn require_pinned_source(val: &str, ctx: &str, stages: &DockerfileStages) -> Result<(), String> {
    let trimmed = val.trim();
    if trimmed.is_empty() || !is_valid_image_or_stage(trimmed, stages, true) {
        return Err(format!(
            "uses unpinned or invalid {ctx} source '{trimmed}'; pin external images to '@sha256:<64-hex>'."
        ));
    }
    Ok(())
}

fn try_parse_directive(
    comment_body: &str,
    escape_char: &mut char,
    seen_escape: &mut bool,
) -> Result<bool, String> {
    let Some((raw_key, raw_val)) = comment_body.split_once('=') else {
        return Ok(false);
    };
    let key = raw_key.trim();
    if key.is_empty()
        || !key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Ok(false);
    }
    if key.eq_ignore_ascii_case("escape") {
        if *seen_escape {
            return Err("duplicate escape directive.".to_string());
        }
        *seen_escape = true;
        match raw_val.trim() {
            "\\" => *escape_char = '\\',
            "`" => *escape_char = '`',
            other => return Err(format!("invalid escape directive '{other}'.")),
        }
    }
    Ok(true)
}

fn fold_dockerfile_instructions(content: &str) -> Result<Vec<String>, String> {
    let content = content.strip_prefix('\u{feff}').unwrap_or(content);
    let (mut escape_char, mut seen_escape) = ('\\', false);
    let (mut in_directives, mut in_continuation) = (true, false);
    let (mut current, mut instructions) = (String::new(), Vec::new());

    for raw_line in content.lines() {
        let trimmed = raw_line.trim();
        if in_directives && !in_continuation {
            if trimmed.is_empty() {
                in_directives = false;
                continue;
            }
            if let Some(body) = trimmed.strip_prefix('#') {
                if try_parse_directive(body, &mut escape_char, &mut seen_escape)? {
                    continue;
                }
                in_directives = false;
                continue;
            }
            in_directives = false;
        }
        if trimmed.starts_with('#') || (!in_continuation && trimmed.is_empty()) {
            continue;
        }
        let trimmed_end = raw_line.trim_end();
        if let Some(without_escape) = trimmed_end.strip_suffix(escape_char) {
            current.push_str(without_escape);
            in_continuation = true;
        } else {
            current.push_str(trimmed_end);
            in_continuation = false;
            if !current.trim().is_empty() {
                instructions.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        }
    }
    if !current.trim().is_empty() {
        instructions.push(current);
    }
    Ok(instructions)
}

fn take_leading_flags(mut rest: &str) -> Result<(Vec<String>, &str), &'static str> {
    let mut flags = Vec::new();
    loop {
        rest = rest.trim_start();
        if !(rest.starts_with("--") || rest.starts_with("\"--") || rest.starts_with("'--")) {
            break;
        }
        let (mut in_quote, mut end_idx, mut cleaned) = (None, rest.len(), String::new());
        for (idx, ch) in rest.char_indices() {
            match (in_quote, ch) {
                (Some(q), c) if c == q => in_quote = None,
                (Some(_), c) => cleaned.push(c),
                (None, '"' | '\'') => in_quote = Some(ch),
                (None, c) if c.is_whitespace() => {
                    end_idx = idx;
                    break;
                }
                (None, c) => cleaned.push(c),
            }
        }
        if in_quote.is_some() {
            return Err("unterminated quote in Dockerfile option");
        }
        rest = &rest[end_idx..];
        if cleaned == "--" {
            break;
        }
        flags.push(cleaned);
    }
    Ok((flags, rest))
}

fn strip_matching_quotes(s: &str) -> &str {
    s.strip_prefix('"')
        .and_then(|r| r.strip_suffix('"'))
        .or_else(|| s.strip_prefix('\'').and_then(|r| r.strip_suffix('\'')))
        .unwrap_or(s)
}

fn validate_from_instruction(rest: &str, stages: &mut DockerfileStages) -> Result<(), String> {
    let (flags, tail) =
        take_leading_flags(rest).map_err(|m| format!("malformed FROM instruction ({m})."))?;
    if let Some(bad) = flags.iter().find(|f| f.ends_with('=')) {
        return Err(format!("malformed FROM flag '{bad}'."));
    }
    let tokens: Vec<&str> = tail.split_whitespace().collect();
    let Some(&raw_spec) = tokens.first() else {
        return Err("malformed FROM instruction.".to_string());
    };
    let image_spec = strip_matching_quotes(raw_spec);
    let alias = match tokens.len() {
        1 => None,
        3 if tokens[1].eq_ignore_ascii_case("as") => {
            let name = strip_matching_quotes(tokens[2]);
            if name.is_empty() || name.contains('$') {
                return Err("malformed FROM stage alias.".to_string());
            }
            Some(name.to_ascii_lowercase())
        }
        _ => return Err("malformed FROM instruction.".to_string()),
    };
    if !is_valid_image_or_stage(image_spec, stages, false) {
        return Err(format!(
            "uses unpinned base image '{image_spec}'; pin base images to '@sha256:<64-hex>'."
        ));
    }
    stages.count += 1;
    if let Some(stage_name) = alias {
        stages.aliases.insert(stage_name);
    }
    Ok(())
}

fn extract_stage_flags(
    instr_name: &str,
    rest: &str,
    stages: &DockerfileStages,
) -> Result<Vec<String>, String> {
    if stages.count == 0 {
        return Err(format!("contains {instr_name} before FROM."));
    }
    take_leading_flags(rest)
        .map(|(flags, _)| flags)
        .map_err(|msg| format!("malformed {instr_name} instruction ({msg})."))
}

fn validate_copy_instruction(rest: &str, stages: &DockerfileStages) -> Result<(), String> {
    for flag in extract_stage_flags("COPY", rest, stages)? {
        if flag.eq_ignore_ascii_case("--from") {
            return Err("empty '--from' flag in COPY.".to_string());
        }
        if let Some((name, raw_val)) = flag.split_once('=')
            && name.eq_ignore_ascii_case("--from")
        {
            require_pinned_source(raw_val, "COPY --from", stages)?;
        }
    }
    Ok(())
}

fn validate_run_mount_spec(mount_spec: &str, stages: &DockerfileStages) -> Result<(), String> {
    let trimmed_spec = mount_spec.trim();
    if trimmed_spec.is_empty() {
        return Err("empty '--mount=' value in RUN.".to_string());
    }
    for field in trimmed_spec.split(',').map(str::trim) {
        if field.eq_ignore_ascii_case("from") {
            return Err("empty 'from' option in RUN --mount.".to_string());
        }
        if let Some((key, val)) = field.split_once('=')
            && key.trim().eq_ignore_ascii_case("from")
        {
            require_pinned_source(val, "RUN --mount from", stages)?;
        }
    }
    Ok(())
}

fn validate_run_instruction(rest: &str, stages: &DockerfileStages) -> Result<(), String> {
    for flag in extract_stage_flags("RUN", rest, stages)? {
        if flag.eq_ignore_ascii_case("--mount") {
            return Err("empty '--mount' flag in RUN.".to_string());
        }
        if let Some((name, raw_val)) = flag.split_once('=')
            && name.eq_ignore_ascii_case("--mount")
        {
            validate_run_mount_spec(raw_val, stages)?;
        }
    }
    Ok(())
}

fn split_instruction_keyword(line: &str) -> Option<(&str, &str)> {
    let keyword = line.split_whitespace().next()?;
    let rest = line
        .trim_start()
        .split_once(char::is_whitespace)
        .map_or("", |(_, r)| r.trim_start());
    Some((keyword, rest))
}

fn validate_dockerfile_instruction(
    line: &str,
    stages: &mut DockerfileStages,
) -> Result<(), String> {
    let (mut current, mut saw_onbuild) = (line, false);
    loop {
        let Some((keyword, rest)) = split_instruction_keyword(current) else {
            return (!saw_onbuild)
                .then_some(())
                .ok_or_else(|| "malformed ONBUILD instruction.".to_string());
        };
        match keyword.to_ascii_lowercase().as_str() {
            "onbuild" if stages.count == 0 => {
                return Err("contains ONBUILD before FROM.".to_string());
            }
            "onbuild" => {
                saw_onbuild = true;
                current = rest;
            }
            "from" if saw_onbuild => return Err("invalid ONBUILD FROM instruction.".to_string()),
            "from" => return validate_from_instruction(rest, stages),
            "copy" => return validate_copy_instruction(rest, stages),
            "run" => return validate_run_instruction(rest, stages),
            _ => return Ok(()),
        }
    }
}

fn validate_dockerfile_pinning(dockerfile_path: &Path, manifest_label: &str) -> Result<(), String> {
    let df = dockerfile_path.display();
    let wrap = |msg: String| format!("Dockerfile '{df}' in '{manifest_label}': {msg}");
    let content =
        fs::read_to_string(dockerfile_path).map_err(|e| wrap(format!("failed to read: {e}")))?;
    let instructions = fold_dockerfile_instructions(&content).map_err(wrap)?;
    let mut stages = DockerfileStages::default();

    for instruction in &instructions {
        validate_dockerfile_instruction(instruction, &mut stages).map_err(wrap)?;
    }
    if stages.count == 0 {
        return Err(wrap("contains no FROM instruction.".to_string()));
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
    let Some(val) = runs_map.get("image") else {
        return (!is_docker).then_some(()).ok_or_else(|| {
            format!("Docker action manifest '{manifest_label}' is missing 'runs.image'.")
        });
    };
    let Some(image_str) = val.as_str() else {
        return Err(format!(
            "Action manifest '{manifest_label}' has a non-string 'runs.image' attribute."
        ));
    };
    let trimmed = image_str.trim();
    if trimmed.is_empty() {
        return Err(format!(
            "Action manifest '{manifest_label}' has an empty 'runs.image' attribute."
        ));
    }
    if trimmed.starts_with("docker://") {
        if !is_pinned_docker_uri(trimmed) {
            return Err(format!(
                "Action manifest '{manifest_label}' has unpinned Docker image '{image_str}'; pin to 'docker://<image>@sha256:<64-hex>'."
            ));
        }
        return Ok(());
    }
    let normalized_ref = if trimmed.starts_with("./") {
        trimmed.to_string()
    } else {
        format!("./{trimmed}")
    };
    let rel_df = extract_safe_local_rel_path(&normalized_ref, false).map_err(|_| {
        format!(
            "Action manifest '{manifest_label}' has invalid or unpinned 'runs.image' '{image_str}'; use 'docker://<image>@sha256:<64-hex>'."
        )
    })?;
    let df_path = action_dir.join(rel_df);
    if !df_path.is_file() {
        return Err(format!(
            "Action manifest '{manifest_label}' has unpinned or missing 'runs.image' '{image_str}'; pin external images to 'docker://<image>@sha256:<64-hex>'."
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
