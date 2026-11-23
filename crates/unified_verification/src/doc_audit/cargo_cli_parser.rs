//! Shell word splitting and `cargo` CLI flag parsing for documentation audits.

pub(super) fn split_shell_words(segment: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;

    for ch in segment.chars() {
        if let Some(q) = quote {
            if ch == q {
                quote = None;
            } else {
                current.push(ch);
            }
        } else if ch == '"' || ch == '\'' {
            quote = Some(ch);
        } else if ch.is_whitespace() {
            if !current.is_empty() {
                words.push(std::mem::take(&mut current));
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

fn is_env_assignment(token: &str) -> bool {
    let Some((key, _)) = token.split_once('=') else {
        return false;
    };
    !key.is_empty() && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

pub(super) fn is_placeholder(token: &str) -> bool {
    (token.contains('<') && token.contains('>')) || token.contains("${{") || token.starts_with('$')
}

#[derive(Default)]
pub(super) struct ParsedCargoCmd {
    pub subcmd: String,
    pub packages: Vec<String>,
    pub features: Vec<String>,
    pub has_feature_placeholder: bool,
    pub example: Option<String>,
    pub bench: Option<String>,
    pub workspace: bool,
    pub no_deps: bool,
    pub all_features: bool,
    pub no_default_features: bool,
    pub lib: bool,
    pub bins: bool,
    pub bin_target: Option<String>,
}

fn push_feature_list(raw: &str, cmd: &mut ParsedCargoCmd) {
    for item in raw
        .split([',', ' '])
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if is_placeholder(item) {
            cmd.has_feature_placeholder = true;
        } else {
            cmd.features.push(item.to_string());
        }
    }
}

pub(super) fn parse_cargo_invocation(tokens: &[String]) -> Option<ParsedCargoCmd> {
    let cargo_idx = tokens.iter().position(|t| !is_env_assignment(t))?;
    if tokens[cargo_idx] != "cargo" {
        return None;
    }
    let rest = &tokens[cargo_idx + 1..];
    let sub_idx = rest.iter().position(|t| !t.starts_with('+'))?;
    let subcmd = rest[sub_idx].clone();
    if !matches!(
        subcmd.as_str(),
        "run" | "test" | "build" | "check" | "bench" | "doc"
    ) {
        return None;
    }

    let mut parsed = ParsedCargoCmd {
        subcmd,
        ..Default::default()
    };
    parse_cargo_flags(&rest[sub_idx + 1..], &mut parsed);
    Some(parsed)
}

fn parse_cargo_flags(args: &[String], parsed: &mut ParsedCargoCmd) {
    let mut idx = 0;
    while idx < args.len() {
        let arg = args[idx].as_str();
        if arg == "--" {
            break;
        }
        if let Some(next_val) = args.get(idx + 1)
            && parse_paired_flag(arg, next_val, parsed)
        {
            idx += 2;
            continue;
        }
        parse_single_flag(arg, parsed);
        idx += 1;
    }
}

fn parse_paired_flag(flag: &str, val: &str, parsed: &mut ParsedCargoCmd) -> bool {
    match flag {
        "-p" | "--package" => parsed.packages.push(val.to_string()),
        "--features" | "-F" => push_feature_list(val, parsed),
        "--example" => parsed.example = Some(val.to_string()),
        "--bench" => parsed.bench = Some(val.to_string()),
        "--bin" => parsed.bin_target = Some(val.to_string()),
        _ => return false,
    }
    true
}

fn parse_single_flag(arg: &str, parsed: &mut ParsedCargoCmd) {
    match arg {
        "--workspace" | "--all" => parsed.workspace = true,
        "--no-deps" => parsed.no_deps = true,
        "--all-features" => parsed.all_features = true,
        "--no-default-features" => parsed.no_default_features = true,
        "--lib" => parsed.lib = true,
        "--bins" => parsed.bins = true,
        _ => parse_equals_flag(arg, parsed),
    }
}

fn parse_equals_flag(arg: &str, parsed: &mut ParsedCargoCmd) {
    if let Some(v) = arg.strip_prefix("--package=") {
        parsed.packages.push(v.to_string());
    } else if let Some(v) = arg.strip_prefix("--features=") {
        push_feature_list(v, parsed);
    } else if let Some(v) = arg.strip_prefix("--example=") {
        parsed.example = Some(v.to_string());
    } else if let Some(v) = arg.strip_prefix("--bench=") {
        parsed.bench = Some(v.to_string());
    }
}
