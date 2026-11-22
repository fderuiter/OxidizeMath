use proc_macro2::TokenStream;
use quote::quote;
use std::fs;

const MAX_EXPANDED_ARGS: usize = 16_384;

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct CrateEnv {
    pub(crate) crate_name: Option<String>,
    pub(crate) has_verified_engine: bool,
    pub(crate) has_scientific_metadata: bool,
    pub(crate) has_math_explorer: bool,
}

fn expand_cli_args<I, S>(raw_args: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut expanded = Vec::new();
    for raw in raw_args {
        if expanded.len() >= MAX_EXPANDED_ARGS {
            break;
        }
        let arg = raw.as_ref();
        if let Some(argfile) = arg.strip_prefix('@')
            && let Ok(contents) = fs::read_to_string(argfile)
        {
            for line in contents.lines() {
                if expanded.len() >= MAX_EXPANDED_ARGS {
                    break;
                }
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    expanded.push(trimmed.to_string());
                }
            }
            continue;
        }
        expanded.push(arg.to_string());
    }
    expanded
}

fn parse_extern_crate_name(spec: &str) -> Option<&str> {
    let before_eq = spec.split('=').next().unwrap_or("");
    let rest = if let Some((opts, after_colon)) = before_eq.split_once(':') {
        if opts.split(',').any(|opt| opt.trim() == "noprelude") {
            return None;
        }
        after_colon
    } else {
        before_eq
    };
    let name = rest.trim();
    (!name.is_empty()).then_some(name)
}

fn record_extern(env: &mut CrateEnv, spec: &str) {
    match parse_extern_crate_name(spec) {
        Some("verified_engine") => env.has_verified_engine = true,
        Some("scientific_metadata") => env.has_scientific_metadata = true,
        Some("math_explorer") => env.has_math_explorer = true,
        _ => {}
    }
}

pub(crate) fn detect_crate_env_from_slice(
    args: &[String],
    cargo_crate_name: Option<&str>,
) -> CrateEnv {
    let mut env = CrateEnv::default();
    let mut idx = 0;
    while idx < args.len() {
        let arg = args[idx].as_str();
        if arg == "--extern" {
            if let Some(next) = args.get(idx + 1) {
                record_extern(&mut env, next);
                idx += 2;
                continue;
            }
        } else if let Some(spec) = arg.strip_prefix("--extern=") {
            record_extern(&mut env, spec);
        } else if arg == "--crate-name" {
            if let Some(next) = args.get(idx + 1) {
                env.crate_name = Some(next.clone());
                idx += 2;
                continue;
            }
        } else if let Some(name) = arg.strip_prefix("--crate-name=") {
            env.crate_name = Some(name.to_string());
        }
        idx += 1;
    }
    if env.crate_name.is_none()
        && let Some(fallback) = cargo_crate_name.filter(|s| !s.is_empty())
    {
        env.crate_name = Some(fallback.replace('-', "_"));
    }
    env
}

fn current_crate_env() -> CrateEnv {
    let args = expand_cli_args(std::env::args());
    let cargo_crate = std::env::var("CARGO_CRATE_NAME").ok();
    detect_crate_env_from_slice(&args, cargo_crate.as_deref())
}

pub(crate) fn verified_engine_path_for(env: &CrateEnv) -> TokenStream {
    if env.has_verified_engine {
        quote! { ::verified_engine }
    } else if env.crate_name.as_deref() == Some("verified_engine") {
        quote! { crate }
    } else if env.has_math_explorer {
        quote! { ::math_explorer::verified_engine }
    } else {
        quote! { ::verified_engine }
    }
}

pub(crate) fn scientific_metadata_path_for(env: &CrateEnv) -> TokenStream {
    if env.has_scientific_metadata {
        quote! { ::scientific_metadata }
    } else if env.has_verified_engine {
        quote! { ::verified_engine::scientific_metadata }
    } else if env.crate_name.as_deref() == Some("verified_engine") {
        quote! { crate::scientific_metadata }
    } else if env.has_math_explorer {
        quote! { ::math_explorer::scientific_metadata }
    } else if env.crate_name.as_deref() == Some("scientific_metadata") {
        quote! { crate }
    } else {
        quote! { ::verified_engine::scientific_metadata }
    }
}

pub(crate) fn resolve_verified_engine_path() -> TokenStream {
    verified_engine_path_for(&current_crate_env())
}

pub(crate) fn resolve_scientific_metadata_path() -> TokenStream {
    scientific_metadata_path_for(&current_crate_env())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn test_resolves_direct_verified_engine_and_internal_crate() {
        let ext = detect_crate_env_from_slice(
            &to_strings(&["rustc", "--extern", "verified_engine=/tmp/libve.rlib"]),
            Some("domain_physics"),
        );
        assert_eq!(
            verified_engine_path_for(&ext).to_string(),
            ":: verified_engine"
        );
        assert_eq!(
            scientific_metadata_path_for(&ext).to_string(),
            ":: verified_engine :: scientific_metadata"
        );

        let self_crate = detect_crate_env_from_slice(
            &to_strings(&["rustc", "--crate-name", "verified_engine"]),
            None,
        );
        assert_eq!(verified_engine_path_for(&self_crate).to_string(), "crate");
        assert_eq!(
            scientific_metadata_path_for(&self_crate).to_string(),
            "crate :: scientific_metadata"
        );
    }

    #[test]
    fn test_resolves_math_explorer_facade_and_noprelude_filtering() {
        let facade = detect_crate_env_from_slice(
            &to_strings(&[
                "rustc",
                "--extern=noprelude:verified_engine=/tmp/libve.rlib",
                "--extern=math_explorer=/tmp/libme.rlib",
            ]),
            None,
        );
        assert!(!facade.has_verified_engine);
        assert!(facade.has_math_explorer);
        assert_eq!(
            verified_engine_path_for(&facade).to_string(),
            ":: math_explorer :: verified_engine"
        );
        assert_eq!(
            scientific_metadata_path_for(&facade).to_string(),
            ":: math_explorer :: scientific_metadata"
        );
    }

    #[test]
    fn test_resolves_scientific_metadata_dev_fallback() {
        let sm_and_ve = detect_crate_env_from_slice(
            &to_strings(&[
                "rustc",
                "--extern",
                "scientific_metadata=/tmp/libsm.rlib",
                "--extern",
                "verified_engine=/tmp/libve.rlib",
            ]),
            None,
        );
        assert_eq!(
            scientific_metadata_path_for(&sm_and_ve).to_string(),
            ":: scientific_metadata"
        );
    }

    #[test]
    fn test_resolves_windows_drive_and_verbatim_extern_paths() {
        let win_facade = detect_crate_env_from_slice(
            &to_strings(&[
                "rustc.exe",
                r"--extern=noprelude:verified_engine=D:\a\OxidizeMath\target\debug\deps\libve.rlib",
                r"--extern=math_explorer=C:\Users\runneradmin\Temp\libme.rlib",
            ]),
            None,
        );
        assert!(!win_facade.has_verified_engine);
        assert!(win_facade.has_math_explorer);
        assert_eq!(
            verified_engine_path_for(&win_facade).to_string(),
            ":: math_explorer :: verified_engine"
        );
        assert_eq!(
            scientific_metadata_path_for(&win_facade).to_string(),
            ":: math_explorer :: scientific_metadata"
        );

        let win_sm = detect_crate_env_from_slice(
            &to_strings(&[
                "rustc.exe",
                "--extern",
                r"scientific_metadata=\\?\D:\a\OxidizeMath\target\debug\libsm.rlib",
                "--extern",
                r"priv:verified_engine=Z:\tmp\libve.rlib",
            ]),
            Some("verified_engine"),
        );
        assert!(win_sm.has_scientific_metadata);
        assert!(win_sm.has_verified_engine);
        assert_eq!(
            verified_engine_path_for(&win_sm).to_string(),
            ":: verified_engine"
        );
        assert_eq!(
            scientific_metadata_path_for(&win_sm).to_string(),
            ":: scientific_metadata"
        );
    }
}
