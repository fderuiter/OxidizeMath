use super::DiffSource;
use super::git_range::{commit_oid, git_stdout, inspect_staged_index, newly_added_files};
use crate::whitelist::{VerificationWhitelist, WHITELIST_FILE_NAME, load_whitelist_strict};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

const ADR_DIR_REL: &str = "docs/adr";
const MIN_ADR_YEAR: u32 = 1970;
const MAX_ADR_YEAR: u32 = 9999;
const MAX_LEGACY_ADR_NUMBER: u16 = 47;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AdrFilename {
    Dated { date: String, slug: String },
    Legacy { prefix: String, slug: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum AdrFilenameError {
    InvalidFormat,
    InvalidCalendarDate(String),
    UnfrozenLegacyPrefix { prefix: String, slug: String },
}

fn is_valid_adr_slug(slug: &str) -> bool {
    !slug.is_empty()
        && slug.split('-').all(|seg| {
            !seg.is_empty()
                && seg
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        })
}

fn is_leap_year(year: u32) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

pub(crate) fn is_valid_yyyymmdd(date_str: &str) -> bool {
    if date_str.len() != 8 || !date_str.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    let Ok(year) = date_str[0..4].parse::<u32>() else {
        return false;
    };
    let Ok(month) = date_str[4..6].parse::<u32>() else {
        return false;
    };
    let Ok(day) = date_str[6..8].parse::<u32>() else {
        return false;
    };
    if !(MIN_ADR_YEAR..=MAX_ADR_YEAR).contains(&year) {
        return false;
    }
    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => return false,
    };
    (1..=max_day).contains(&day)
}

fn parse_adr_rel_path(rel_path: &str) -> Result<AdrFilename, AdrFilenameError> {
    let normalized = rel_path.replace('\\', "/");
    let Some(filename) = normalized.strip_prefix("docs/adr/") else {
        return Err(AdrFilenameError::InvalidFormat);
    };
    if filename.contains('/') {
        return Err(AdrFilenameError::InvalidFormat);
    }
    let Some(stem) = filename.strip_suffix(".md") else {
        return Err(AdrFilenameError::InvalidFormat);
    };
    let Some((prefix, slug)) = stem.split_once('-') else {
        return Err(AdrFilenameError::InvalidFormat);
    };
    if !is_valid_adr_slug(slug) {
        return Err(AdrFilenameError::InvalidFormat);
    }

    if prefix.len() == 8 && prefix.bytes().all(|b| b.is_ascii_digit()) {
        if is_valid_yyyymmdd(prefix) {
            return Ok(AdrFilename::Dated {
                date: prefix.to_string(),
                slug: slug.to_string(),
            });
        }
        return Err(AdrFilenameError::InvalidCalendarDate(prefix.to_string()));
    }

    if prefix.len() == 4
        && prefix.bytes().all(|b| b.is_ascii_digit())
        && let Ok(num) = prefix.parse::<u16>()
    {
        if (1..=MAX_LEGACY_ADR_NUMBER).contains(&num) {
            return Ok(AdrFilename::Legacy {
                prefix: prefix.to_string(),
                slug: slug.to_string(),
            });
        }
        return Err(AdrFilenameError::UnfrozenLegacyPrefix {
            prefix: prefix.to_string(),
            slug: slug.to_string(),
        });
    }

    Err(AdrFilenameError::InvalidFormat)
}

pub(crate) fn is_valid_adr_rel_path(rel_path: &str) -> bool {
    parse_adr_rel_path(rel_path).is_ok()
}

fn extract_four_digit_adr_slug(rel_path: &str) -> Option<&str> {
    let filename = rel_path.strip_prefix("docs/adr/")?;
    if filename.contains('/') {
        return None;
    }
    let stem = filename.strip_suffix(".md")?;
    let (prefix, slug) = stem.split_once('-')?;
    (prefix.len() == 4 && prefix.bytes().all(|b| b.is_ascii_digit()) && !slug.is_empty())
        .then_some(slug)
}

fn normalize_allowlist_entry(entry: &str) -> String {
    let clean = entry.trim().replace('\\', "/");
    let trimmed = clean.trim_start_matches("./");
    if trimmed.contains('/') {
        trimmed.to_string()
    } else {
        format!("{ADR_DIR_REL}/{trimmed}")
    }
}

fn load_adr_allowlist(repo_dir: &Path, source: &DiffSource) -> Result<HashSet<String>, String> {
    let parsed: VerificationWhitelist = if matches!(source, DiffSource::Staged)
        && let Some(staged_toml) =
            git_stdout(repo_dir, &["show", &format!(":{WHITELIST_FILE_NAME}")])
    {
        toml::from_str(&staged_toml)
            .map_err(|err| format!("Failed to parse staged {WHITELIST_FILE_NAME}: {err}"))?
    } else {
        let primary = repo_dir.join(WHITELIST_FILE_NAME);
        if primary.exists() {
            load_whitelist_strict(Some(&primary), &[] as &[&Path])?
        } else if repo_dir == Path::new(".")
            && Path::new("../../verification_whitelist.toml").exists()
        {
            load_whitelist_strict(
                Some(Path::new("../../verification_whitelist.toml")),
                &[] as &[&Path],
            )?
        } else {
            VerificationWhitelist::default()
        }
    };

    let entries = parsed
        .adr_verification
        .and_then(|cfg| cfg.allowlist)
        .unwrap_or_default()
        .into_iter()
        .map(|s| normalize_allowlist_entry(&s))
        .collect();
    Ok(entries)
}

fn collect_staged_adrs(repo_dir: &Path) -> Result<BTreeMap<String, String>, String> {
    let mut files = BTreeMap::new();
    let Some(out) = git_stdout(repo_dir, &["ls-files", "--", ADR_DIR_REL]) else {
        return Ok(files);
    };
    for raw in out.lines() {
        let rel = raw.trim().replace('\\', "/");
        if rel.is_empty() {
            continue;
        }
        let content = git_stdout(repo_dir, &["show", &format!(":{rel}")])
            .ok_or_else(|| format!("Failed to read staged blob for {rel}"))?;
        files.insert(rel, content);
    }
    Ok(files)
}

fn collect_commit_tree_adrs(
    repo_dir: &Path,
    commit: &str,
) -> Result<BTreeMap<String, String>, String> {
    let mut files = BTreeMap::new();
    let Some(out) = git_stdout(
        repo_dir,
        &["ls-tree", "-r", "--name-only", commit, "--", ADR_DIR_REL],
    ) else {
        return Ok(files);
    };
    for raw in out.lines() {
        let rel = raw.trim().replace('\\', "/");
        if rel.is_empty() {
            continue;
        }
        let content = git_stdout(repo_dir, &["show", &format!("{commit}:{rel}")])
            .ok_or_else(|| format!("Failed to read {commit}:{rel}"))?;
        files.insert(rel, content);
    }
    Ok(files)
}

fn collect_working_tree_adrs(
    repo_dir: &Path,
    prefer_staged_blobs: bool,
) -> Result<BTreeMap<String, String>, String> {
    let mut files = BTreeMap::new();
    let staged_set: HashSet<String> = if prefer_staged_blobs {
        inspect_staged_index(repo_dir)
            .lines()
            .map(|l| l.trim().replace('\\', "/"))
            .filter(|l| !l.is_empty())
            .collect()
    } else {
        HashSet::new()
    };

    let adr_root = repo_dir.join(ADR_DIR_REL);
    if adr_root.is_dir() {
        for entry in WalkDir::new(&adr_root).into_iter().filter_map(Result::ok) {
            if !entry.file_type().is_file() {
                continue;
            }
            let rel_os = entry.path().strip_prefix(repo_dir).unwrap_or(entry.path());
            let rel = rel_os.to_string_lossy().replace('\\', "/");
            let rel = rel.trim_start_matches("./").to_string();
            let content = if staged_set.contains(&rel)
                && let Some(blob) = git_stdout(repo_dir, &["show", &format!(":{rel}")])
            {
                blob
            } else {
                fs::read_to_string(entry.path())
                    .map_err(|e| format!("Failed to read {}: {e}", entry.path().display()))?
            };
            files.insert(rel, content);
        }
    }

    if prefer_staged_blobs {
        for rel in &staged_set {
            if rel.starts_with("docs/adr/")
                && !files.contains_key(rel)
                && let Some(blob) = git_stdout(repo_dir, &["show", &format!(":{rel}")])
            {
                files.insert(rel.clone(), blob);
            }
        }
    }
    Ok(files)
}

fn collect_adrs_for_source(
    repo_dir: &Path,
    source: &DiffSource,
) -> Result<BTreeMap<String, String>, String> {
    match source {
        DiffSource::Staged => collect_staged_adrs(repo_dir),
        DiffSource::Range { head, .. }
            if commit_oid(repo_dir, head) != commit_oid(repo_dir, "HEAD") =>
        {
            collect_commit_tree_adrs(repo_dir, head)
        }
        DiffSource::CachedBase(_) => collect_working_tree_adrs(repo_dir, true),
        DiffSource::Range { .. } | DiffSource::Head => collect_working_tree_adrs(repo_dir, false),
    }
}

fn update_fence_state(trimmed_start: &str, active_fence: &mut Option<(char, usize)>) -> bool {
    if let Some((fence_ch, min_len)) = *active_fence {
        let run = trimmed_start.chars().take_while(|&c| c == fence_ch).count();
        if run >= min_len && trimmed_start[run..].trim().is_empty() {
            *active_fence = None;
        }
        return true;
    }

    let backticks = trimmed_start.chars().take_while(|&c| c == '`').count();
    if backticks >= 3 && !trimmed_start[backticks..].contains('`') {
        *active_fence = Some(('`', backticks));
        return true;
    }

    let tildes = trimmed_start.chars().take_while(|&c| c == '~').count();
    if tildes >= 3 {
        *active_fence = Some(('~', tildes));
        return true;
    }
    false
}

fn is_status_heading(line: &str) -> bool {
    let trimmed = line.trim_end();
    trimmed == "## Status"
        || trimmed
            .strip_prefix("## Status")
            .is_some_and(|rest| rest.trim_matches('#').trim().is_empty())
}

pub(crate) fn check_markdown_headings(content: &str) -> (bool, bool) {
    let mut active_fence: Option<(char, usize)> = None;
    let mut has_title = false;
    let mut has_status = false;

    for (idx, raw_line) in content.lines().enumerate() {
        let line = if idx == 0 {
            raw_line.strip_prefix('\u{feff}').unwrap_or(raw_line)
        } else {
            raw_line
        };
        if update_fence_state(line.trim_start(), &mut active_fence) {
            continue;
        }
        if line
            .strip_prefix("# ")
            .is_some_and(|rest| !rest.trim().is_empty())
        {
            has_title = true;
        } else if is_status_heading(line) {
            has_status = true;
        }
    }
    (has_title, has_status)
}

fn check_heading_violations(rel_path: &str, content: &str, violations: &mut Vec<String>) {
    let (has_title, has_status) = check_markdown_headings(content);
    if !has_title {
        violations.push(format!(
            "{rel_path}: Missing top-level '# <Title>' heading outside code fences"
        ));
    }
    if !has_status {
        violations.push(format!(
            "{rel_path}: Missing required '## Status' heading outside code fences"
        ));
    }
}

fn append_collision_violations(
    by_dated_slug: BTreeMap<String, Vec<String>>,
    by_legacy_prefix: BTreeMap<String, Vec<String>>,
    allowlist: &HashSet<String>,
    violations: &mut Vec<String>,
) {
    for (slug, paths) in by_dated_slug {
        if paths.len() > 1 {
            violations.push(format!(
                "Duplicate ADR slug '{slug}' across [{}]; each dated ADR must use a unique <slug>",
                paths.join(", ")
            ));
        }
    }

    for (prefix, paths) in by_legacy_prefix {
        if paths.len() > 1 {
            let unauthorized: Vec<&str> = paths
                .iter()
                .map(String::as_str)
                .filter(|p| !allowlist.contains(*p))
                .collect();
            if !unauthorized.is_empty() {
                violations.push(format!(
                    "Duplicate ADR prefix '{prefix}' across [{}] (unallowlisted: [{}])",
                    paths.join(", "),
                    unauthorized.join(", ")
                ));
            }
        }
    }
}

fn collect_adr_violations(
    adrs: &BTreeMap<String, String>,
    allowlist: &HashSet<String>,
) -> Vec<String> {
    let mut violations = Vec::new();
    let mut by_legacy_prefix: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut by_dated_slug: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for (rel_path, content) in adrs {
        match parse_adr_rel_path(rel_path) {
            Ok(AdrFilename::Dated { slug, .. }) => {
                by_dated_slug
                    .entry(slug)
                    .or_default()
                    .push(rel_path.clone());
            }
            Ok(AdrFilename::Legacy { prefix, .. }) => {
                by_legacy_prefix
                    .entry(prefix)
                    .or_default()
                    .push(rel_path.clone());
            }
            Err(AdrFilenameError::InvalidCalendarDate(date)) => {
                violations.push(format!(
                    "{rel_path}: Invalid calendar date '{date}' in ADR filename; expected valid docs/adr/YYYYMMDD-slug.md"
                ));
                continue;
            }
            Err(AdrFilenameError::UnfrozenLegacyPrefix { prefix, slug }) => {
                violations.push(format!(
                    "{rel_path}: Legacy 4-digit ADR prefix '{prefix}' is frozen (0001..={MAX_LEGACY_ADR_NUMBER:04}); rename to docs/adr/YYYYMMDD-{slug}.md"
                ));
                continue;
            }
            Err(AdrFilenameError::InvalidFormat) => {
                violations.push(format!(
                    "{rel_path}: ADR filename must match docs/adr/YYYYMMDD-slug.md (or frozen legacy docs/adr/NNNN-slug.md)"
                ));
                continue;
            }
        }
        check_heading_violations(rel_path, content, &mut violations);
    }

    append_collision_violations(by_dated_slug, by_legacy_prefix, allowlist, &mut violations);
    violations
}

pub(crate) fn validate_no_new_legacy_adrs(repo_dir: &Path, source: &DiffSource) -> bool {
    let added = newly_added_files(repo_dir, source);
    let mut violations = Vec::new();
    for rel_path in added {
        if let Some(slug) = extract_four_digit_adr_slug(&rel_path) {
            violations.push(format!(
                "{rel_path}: Newly added 4-digit ADR filenames (NNNN-slug.md) are rejected because legacy prefixes are frozen; rename to docs/adr/YYYYMMDD-{slug}.md"
            ));
        }
    }
    if !violations.is_empty() {
        println!("ADR structural/numbering verification failed:");
        for v in &violations {
            println!("  - {v}");
        }
        return false;
    }
    true
}

pub(crate) fn validate_repository_adrs(repo_dir: &Path, source: &DiffSource) -> bool {
    let allowlist = match load_adr_allowlist(repo_dir, source) {
        Ok(list) => list,
        Err(err) => {
            eprintln!("ADR verification failed: {err}");
            return false;
        }
    };

    let adrs = match collect_adrs_for_source(repo_dir, source) {
        Ok(map) => map,
        Err(err) => {
            eprintln!("ADR verification failed: {err}");
            return false;
        }
    };

    let violations = collect_adr_violations(&adrs, &allowlist);
    if !violations.is_empty() {
        println!("ADR structural/numbering verification failed:");
        for v in &violations {
            println!("  - {v}");
        }
        return false;
    }
    true
}
