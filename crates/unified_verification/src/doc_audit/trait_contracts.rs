use regex::Regex;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use syn::visit::Visit;

#[derive(Debug, Clone, PartialEq, Eq)]
struct NormalizedSig {
    receiver: &'static str,
    params: Vec<String>,
    ret: String,
    has_default: bool,
}

#[derive(Debug, Clone, Default)]
pub struct TraitContractRegistry {
    traits: BTreeMap<String, BTreeMap<String, NormalizedSig>>,
}

fn normalize_receiver(sig: &syn::Signature) -> &'static str {
    let Some(syn::FnArg::Receiver(recv)) = sig.inputs.first() else {
        return "none";
    };
    match (&recv.reference, &recv.mutability) {
        (Some(_), Some(_)) => "&mut self",
        (Some(_), None) => "&self",
        (None, Some(_)) => "mut self",
        (None, None) => "self",
    }
}

fn normalize_type(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Reference(r) => {
            let mut_kw = if r.mutability.is_some() { "mut " } else { "" };
            format!("&{}{}", mut_kw, normalize_type(&r.elem))
        }
        syn::Type::Slice(s) => format!("[{}]", normalize_type(&s.elem)),
        syn::Type::Tuple(t) => {
            let inner: Vec<String> = t.elems.iter().map(normalize_type).collect();
            format!("({})", inner.join(","))
        }
        syn::Type::TraitObject(obj) => {
            let bounds: Vec<String> = obj
                .bounds
                .iter()
                .filter_map(|b| match b {
                    syn::TypeParamBound::Trait(tb) => {
                        tb.path.segments.last().map(|s| s.ident.to_string())
                    }
                    _ => None,
                })
                .collect();
            format!("dyn {}", bounds.join("+"))
        }
        syn::Type::Path(tp) => {
            let Some(last) = tp.path.segments.last() else {
                return String::new();
            };
            let ident = last.ident.to_string();
            match &last.arguments {
                syn::PathArguments::None => ident,
                syn::PathArguments::AngleBracketed(ab) => {
                    let args: Vec<String> = ab
                        .args
                        .iter()
                        .filter_map(|ga| match ga {
                            syn::GenericArgument::Type(inner) => Some(normalize_type(inner)),
                            _ => None,
                        })
                        .collect();
                    if args.is_empty() {
                        ident
                    } else {
                        format!("{ident}<{}>", args.join(","))
                    }
                }
                syn::PathArguments::Parenthesized(_) => ident,
            }
        }
        other => quote::quote!(#other).to_string().replace(' ', ""),
    }
}

fn normalize_signature(sig: &syn::Signature, has_default: bool) -> NormalizedSig {
    let params = sig
        .inputs
        .iter()
        .filter_map(|arg| match arg {
            syn::FnArg::Receiver(_) => None,
            syn::FnArg::Typed(pat_ty) => Some(normalize_type(&pat_ty.ty)),
        })
        .collect();
    let ret = match &sig.output {
        syn::ReturnType::Default => "()".to_string(),
        syn::ReturnType::Type(_, ty) => normalize_type(ty),
    };
    NormalizedSig {
        receiver: normalize_receiver(sig),
        params,
        ret,
        has_default,
    }
}

impl TraitContractRegistry {
    /// Parses canonical trait contracts (`TheoryDescribable`, `InteractiveTool`, `ExplorerTab`) from workspace sources.
    pub fn load(workspace_root: &Path) -> Self {
        let mut reg = Self::default();
        for (rel_path, trait_name) in [
            (
                "crates/scientific_metadata/src/theory.rs",
                "TheoryDescribable",
            ),
            ("math_explorer_gui/src/framework/mod.rs", "InteractiveTool"),
            ("math_explorer_gui/src/tabs/mod.rs", "ExplorerTab"),
        ] {
            let full_path = workspace_root.join(rel_path);
            if let Ok(src) = fs::read_to_string(&full_path)
                && let Ok(ast) = syn::parse_file(&src)
                && let Some(methods) = extract_trait_methods(&ast, trait_name)
            {
                reg.traits.insert(trait_name.to_string(), methods);
            }
        }
        reg
    }
}

fn extract_trait_methods(
    ast: &syn::File,
    target_name: &str,
) -> Option<BTreeMap<String, NormalizedSig>> {
    for item in &ast.items {
        let syn::Item::Trait(item_trait) = item else {
            continue;
        };
        if item_trait.ident != target_name {
            continue;
        }
        let mut methods = BTreeMap::new();
        for t_item in &item_trait.items {
            if let syn::TraitItem::Fn(m) = t_item {
                methods.insert(
                    m.sig.ident.to_string(),
                    normalize_signature(&m.sig, m.default.is_some()),
                );
            }
        }
        return Some(methods);
    }
    None
}

struct ImplCollector<'a> {
    path: &'a Path,
    fence_line: usize,
    has_elision: bool,
    registry: &'a TraitContractRegistry,
    violations: Vec<String>,
}

impl<'ast> Visit<'ast> for ImplCollector<'_> {
    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        if let Some((_, trait_path, _)) = &node.trait_
            && let Some(last_seg) = trait_path.segments.last()
        {
            let trait_name = last_seg.ident.to_string();
            if let Some(expected_methods) = self.registry.traits.get(&trait_name) {
                validate_impl_against_trait(
                    self.path,
                    self.fence_line,
                    &trait_name,
                    node,
                    expected_methods,
                    self.has_elision,
                    &mut self.violations,
                );
            }
        }
        syn::visit::visit_item_impl(self, node);
    }
}

fn validate_impl_against_trait(
    path: &Path,
    fence_line: usize,
    trait_name: &str,
    node: &syn::ItemImpl,
    expected_methods: &BTreeMap<String, NormalizedSig>,
    has_elision: bool,
    violations: &mut Vec<String>,
) {
    let mut implemented = BTreeSet::new();
    for item in &node.items {
        let syn::ImplItem::Fn(method) = item else {
            continue;
        };
        let m_name = method.sig.ident.to_string();
        implemented.insert(m_name.clone());
        let Some(expected) = expected_methods.get(&m_name) else {
            violations.push(format!(
                "{}:{}: `impl {trait_name}` defines non-existent trait method `{m_name}`",
                path.display(),
                fence_line
            ));
            continue;
        };
        let actual = normalize_signature(&method.sig, false);
        if actual.receiver != expected.receiver
            || actual.params != expected.params
            || actual.ret != expected.ret
        {
            violations.push(format!(
                "{}:{}: `impl {trait_name}::{m_name}` signature mismatch (expected `({}) -> {}`, found `({}) -> {}`)",
                path.display(),
                fence_line,
                format_sig_inputs(expected),
                expected.ret,
                format_sig_inputs(&actual),
                actual.ret
            ));
        }
    }

    if !has_elision {
        for (req_name, sig) in expected_methods {
            if !sig.has_default && !implemented.contains(req_name) {
                violations.push(format!(
                    "{}:{}: `impl {trait_name}` omits required trait method `{req_name}`",
                    path.display(),
                    fence_line
                ));
            }
        }
    }
}

fn format_sig_inputs(sig: &NormalizedSig) -> String {
    let mut parts = Vec::new();
    if sig.receiver != "none" {
        parts.push(sig.receiver.to_string());
    }
    parts.extend(sig.params.iter().cloned());
    parts.join(", ")
}

fn parse_fence_ast(raw_block: &str) -> Result<syn::File, syn::Error> {
    let mut cleaned = String::new();
    for line in raw_block.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("# ") {
            cleaned.push_str(rest);
        } else if trimmed != "#" {
            cleaned.push_str(line);
        }
        cleaned.push('\n');
    }
    syn::parse_file(&cleaned)
        .or_else(|_| syn::parse_file(&format!("fn __doctest_wrap() {{\n{cleaned}\n}}")))
}

fn mentions_tracked_trait_impl(block: &str) -> bool {
    ["TheoryDescribable", "InteractiveTool", "ExplorerTab"]
        .iter()
        .any(|t| block.contains(&format!("{t} for ")))
}

/// Validates trait `impl` blocks inside Markdown code fences (including `rust,ignore`) and GUI onboarding invariants.
pub fn check_markdown_trait_and_gui_contracts(
    md_path: &Path,
    content: &str,
    registry: &TraitContractRegistry,
) -> Vec<String> {
    let mut violations = Vec::new();
    if !registry.traits.is_empty() {
        check_fence_trait_impls(md_path, content, registry, &mut violations);
    }
    check_gui_guide_invariants(md_path, content, &mut violations);
    violations
}

fn check_fence_trait_impls(
    md_path: &Path,
    content: &str,
    registry: &TraitContractRegistry,
    violations: &mut Vec<String>,
) {
    let mut in_fence = false;
    let mut fence_start = 1usize;
    let mut block = String::new();

    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            if !in_fence {
                in_fence = true;
                fence_start = idx + 1;
                block.clear();
            } else {
                if mentions_tracked_trait_impl(&block) {
                    inspect_trait_fence(md_path, fence_start, &block, registry, violations);
                }
                in_fence = false;
                block.clear();
            }
            continue;
        }
        if in_fence {
            block.push_str(line);
            block.push('\n');
        }
    }
}

fn inspect_trait_fence(
    md_path: &Path,
    fence_line: usize,
    block: &str,
    registry: &TraitContractRegistry,
    violations: &mut Vec<String>,
) {
    match parse_fence_ast(block) {
        Ok(ast) => {
            let mut collector = ImplCollector {
                path: md_path,
                fence_line,
                has_elision: block.contains("// ..."),
                registry,
                violations: Vec::new(),
            };
            collector.visit_file(&ast);
            violations.extend(collector.violations);
        }
        Err(err) => violations.push(format!(
            "{}:{}: Unparseable trait `impl` snippet in guide: {err}",
            md_path.display(),
            fence_line
        )),
    }
}

fn check_gui_guide_invariants(md_path: &Path, content: &str, violations: &mut Vec<String>) {
    let norm = md_path.to_string_lossy().replace('\\', "/");
    let is_gui_guide = norm.ends_with("math_explorer_gui/README.md")
        || norm.ends_with("docs/architecture/interactive_tool_walkthrough.md");
    if !is_gui_guide {
        return;
    }

    let app_rs_re = Regex::new(r"\bsrc/app\.rs\b").expect("Valid regex");
    let manual_tabs_mod_re = Regex::new(
        r"(?i)(?:in\s+`?(?:math_explorer_gui/)?src/tabs/mod\.rs`?,\s*add|register\s+the\s+module\s+inside\s+`?(?:math_explorer_gui/)?src/tabs/mod\.rs`?)",
    )
    .expect("Valid regex");

    for (idx, line) in content.lines().enumerate() {
        if app_rs_re.is_match(line) {
            violations.push(format!(
                "{}:{}: GUI guide references non-existent `src/app.rs` (`build.rs` auto-discovers tabs)",
                md_path.display(),
                idx + 1
            ));
        }
        if manual_tabs_mod_re.is_match(line) {
            violations.push(format!(
                "{}:{}: GUI guide instructs manual edits to `src/tabs/mod.rs` (`build.rs` auto-generates tab registration)",
                md_path.display(),
                idx + 1
            ));
        }
    }

    if content.contains("fn step_simulation")
        && (content.contains("0.016")
            || (!content.contains("GIF") && !content.contains("sample_animation_frames")))
    {
        violations.push(format!(
            "{}: `step_simulation` must be documented as an offscreen GIF export hook (`sample_animation_frames`), not a 0.016s display frame loop",
            md_path.display()
        ));
    }
}
