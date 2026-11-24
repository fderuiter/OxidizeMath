use oxidize_core::bibtex::{
    BibtexDatabase, extract_doc_bib_keys, extract_doc_spec_ids, validate_math_spec_markdown,
};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use syn::visit::Visit;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractViolation {
    pub line: usize,
    pub item_label: String,
    pub missing_parts: Vec<String>,
}

pub(crate) struct MathDocContractVisitor<'a> {
    workspace_root: &'a Path,
    bib_db: &'a BibtexDatabase,
    current_impl_type: Option<String>,
    current_trait_name: Option<String>,
    pub(crate) violations: Vec<ContractViolation>,
}

impl<'a> MathDocContractVisitor<'a> {
    pub(crate) fn new(workspace_root: &'a Path, bib_db: &'a BibtexDatabase) -> Self {
        Self {
            workspace_root,
            bib_db,
            current_impl_type: None,
            current_trait_name: None,
            violations: Vec::new(),
        }
    }

    fn audit_item(
        &mut self,
        attrs: &[syn::Attribute],
        line: usize,
        item_label: String,
        allow_theory_derive: bool,
    ) {
        let doc_prose = extract_outer_doc_prose(attrs);
        let lower_prose = doc_prose.to_ascii_lowercase();
        let (has_embed_theory, embed_papers) = extract_embed_theory_papers(attrs);
        let (is_theory_struct, theory_citation) = if allow_theory_derive {
            extract_theory_derive_citation(attrs)
        } else {
            (false, None)
        };

        let mut missing = Vec::new();
        if !has_latex_equation(&doc_prose, has_embed_theory) {
            missing.push("LaTeX equation (`$$...$$`, `$...$`, or `#[embed_theory]`)".to_string());
        }
        missing.extend(check_citations_and_specs(
            self.workspace_root,
            self.bib_db,
            &doc_prose,
            &embed_papers,
            is_theory_struct,
            theory_citation.as_deref(),
        ));
        if !has_stability_note(&lower_prose) {
            missing.push("numerical stability / conditioning / error bounds".to_string());
        }
        if !has_validity_domain(&lower_prose) {
            missing.push("domain of validity / preconditions".to_string());
        }

        if !missing.is_empty() {
            self.violations.push(ContractViolation {
                line,
                item_label,
                missing_parts: missing,
            });
        }
    }
}

impl<'ast> Visit<'ast> for MathDocContractVisitor<'_> {
    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        if attrs_skip_item(&i.attrs) {
            return;
        }
        if i.content.is_some() && !matches!(i.vis, syn::Visibility::Public(_)) {
            return;
        }
        syn::visit::visit_item_mod(self, i);
    }

    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        if attrs_skip_item(&i.attrs) {
            return;
        }
        if matches!(i.vis, syn::Visibility::Public(_)) && i.sig.ident != "main" {
            self.audit_item(
                &i.attrs,
                i.sig.ident.span().start().line,
                format!("fn `{}`", i.sig.ident),
                false,
            );
        }
    }

    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        if attrs_skip_item(&i.attrs) {
            return;
        }
        if matches!(i.vis, syn::Visibility::Public(_)) {
            self.audit_item(
                &i.attrs,
                i.ident.span().start().line,
                format!("struct `{}`", i.ident),
                true,
            );
        }
    }

    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        if attrs_skip_item(&i.attrs) {
            return;
        }
        if matches!(i.vis, syn::Visibility::Public(_)) {
            self.audit_item(
                &i.attrs,
                i.ident.span().start().line,
                format!("enum `{}`", i.ident),
                false,
            );
        }
    }

    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        if attrs_skip_item(&i.attrs) {
            return;
        }
        if matches!(i.vis, syn::Visibility::Public(_)) {
            let trait_name = i.ident.to_string();
            self.audit_item(
                &i.attrs,
                i.ident.span().start().line,
                format!("trait `{trait_name}`"),
                false,
            );
            let prev = self.current_trait_name.replace(trait_name);
            syn::visit::visit_item_trait(self, i);
            self.current_trait_name = prev;
        }
    }

    fn visit_trait_item_fn(&mut self, i: &'ast syn::TraitItemFn) {
        if attrs_skip_item(&i.attrs) {
            return;
        }
        if let Some(trait_name) = self.current_trait_name.clone() {
            self.audit_item(
                &i.attrs,
                i.sig.ident.span().start().line,
                format!("trait method `{trait_name}::{}`", i.sig.ident),
                false,
            );
        }
    }

    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        if attrs_skip_item(&i.attrs) || i.trait_.is_some() {
            return;
        }
        let ty_name =
            crate::ast_helpers::impl_base_ident(&i.self_ty).unwrap_or_else(|| "<impl>".to_string());
        let prev = self.current_impl_type.replace(ty_name);
        syn::visit::visit_item_impl(self, i);
        self.current_impl_type = prev;
    }

    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        if attrs_skip_item(&i.attrs) {
            return;
        }
        if matches!(i.vis, syn::Visibility::Public(_)) {
            let owner = self.current_impl_type.as_deref().unwrap_or("<impl>");
            self.audit_item(
                &i.attrs,
                i.sig.ident.span().start().line,
                format!("method `{owner}::{}`", i.sig.ident),
                false,
            );
        }
    }
}

fn attrs_skip_item(attrs: &[syn::Attribute]) -> bool {
    for attr in attrs {
        if attr.path().is_ident("cfg")
            && let Ok(meta) = attr.parse_args::<syn::Meta>()
            && meta_contains_test_cfg(&meta)
        {
            return true;
        }
        if attr.path().is_ident("doc")
            && let Ok(nested) = attr.parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            )
            && nested.iter().any(|m| m.path().is_ident("hidden"))
        {
            return true;
        }
    }
    false
}

fn meta_contains_test_cfg(meta: &syn::Meta) -> bool {
    match meta {
        syn::Meta::Path(p) => p.is_ident("test"),
        syn::Meta::List(list) if list.path.is_ident("all") || list.path.is_ident("any") => list
            .parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            )
            .is_ok_and(|nested| nested.iter().any(meta_contains_test_cfg)),
        _ => false,
    }
}

fn extract_outer_doc_prose(attrs: &[syn::Attribute]) -> String {
    let mut raw_lines = Vec::new();
    for attr in attrs {
        if !matches!(attr.style, syn::AttrStyle::Outer) || !attr.path().is_ident("doc") {
            continue;
        }
        if let syn::Meta::NameValue(nv) = &attr.meta
            && let syn::Expr::Lit(expr_lit) = &nv.value
            && let syn::Lit::Str(s) = &expr_lit.lit
        {
            raw_lines.push(s.value());
        }
    }
    let joined = raw_lines.join("\n");
    let mut prose = Vec::new();
    let mut in_fence = false;
    for line in joined.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if !in_fence {
            prose.push(line);
        }
    }
    prose.join("\n")
}

fn extract_embed_theory_papers(attrs: &[syn::Attribute]) -> (bool, Vec<String>) {
    let mut has_embed = false;
    let mut papers = Vec::new();
    for attr in attrs {
        let is_embed = attr
            .path()
            .segments
            .last()
            .is_some_and(|seg| seg.ident == "embed_theory");
        if !is_embed {
            continue;
        }
        has_embed = true;
        if let Ok(exprs) = attr.parse_args_with(
            syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated,
        ) && let Some(syn::Expr::Lit(el)) = exprs.first()
            && let syn::Lit::Str(s) = &el.lit
        {
            let val = s.value();
            if val.starts_with("papers/") && val.ends_with(".tex") {
                papers.push(val);
            }
        }
    }
    (has_embed, papers)
}

fn extract_theory_derive_citation(attrs: &[syn::Attribute]) -> (bool, Option<String>) {
    let mut is_theory = false;
    let mut citation = None;
    for attr in attrs {
        if attr.path().is_ident("derive")
            && let Ok(paths) = attr.parse_args_with(
                syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
            )
            && paths
                .iter()
                .any(|p| p.segments.last().is_some_and(|s| s.ident == "Theory"))
        {
            is_theory = true;
        }
        if attr.path().is_ident("theory")
            && let Ok(metas) = attr.parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            )
        {
            for m in metas {
                if let syn::Meta::NameValue(nv) = m
                    && nv.path.is_ident("citation")
                    && let syn::Expr::Lit(el) = nv.value
                    && let syn::Lit::Str(s) = el.lit
                {
                    citation = Some(s.value());
                }
            }
        }
    }
    (is_theory, citation)
}

fn has_latex_equation(prose: &str, has_embed_theory: bool) -> bool {
    if has_embed_theory
        || prose.contains("$$")
        || prose.contains("\\[")
        || prose.contains("\\begin{equation}")
        || prose.contains("\\begin{align}")
    {
        return true;
    }
    prose.lines().any(line_has_inline_math)
}

fn line_has_inline_math(line: &str) -> bool {
    let mut outside_code = String::with_capacity(line.len());
    let mut in_ticks = false;
    for ch in line.chars() {
        if ch == '`' {
            in_ticks = !in_ticks;
        } else if !in_ticks {
            outside_code.push(ch);
        }
    }
    let bytes = outside_code.as_bytes();
    let mut idx = 0usize;
    while idx < bytes.len() {
        if bytes[idx] == b'$' {
            let mut end = idx + 1;
            while end < bytes.len() && bytes[end] != b'$' {
                end += 1;
            }
            if end < bytes.len()
                && outside_code[idx + 1..end]
                    .chars()
                    .any(|c| !c.is_whitespace())
            {
                return true;
            }
            idx = end.saturating_add(1);
        } else {
            idx += 1;
        }
    }
    false
}

fn check_citations_and_specs(
    workspace_root: &Path,
    bib_db: &BibtexDatabase,
    prose: &str,
    embed_papers: &[String],
    is_theory_struct: bool,
    theory_citation: Option<&str>,
) -> Vec<String> {
    let mut issues = Vec::new();
    let doc_keys = extract_doc_bib_keys(prose);
    if doc_keys.is_empty() {
        issues.push("verified BibTeX key in docstring (`papers/references.bib`)".to_string());
    } else {
        for key in &doc_keys {
            if let Err(err) = bib_db.verify_cited_key(key) {
                issues.push(format!("verified BibTeX key `{key}` ({err})"));
            }
        }
    }

    if is_theory_struct {
        match theory_citation {
            None => issues.push("#[theory(citation = \"<bib_key>\")] attribute".to_string()),
            Some(raw) => {
                let keys: Vec<&str> = raw
                    .split([',', ';'])
                    .map(str::trim)
                    .filter(|k| !k.is_empty())
                    .collect();
                if keys.is_empty() {
                    issues.push("non-empty BibTeX key in #[theory(citation = ...)]".to_string());
                }
                for key in keys {
                    if let Err(err) = bib_db.verify_cited_key(key) {
                        issues.push(format!("#[theory(citation)] key `{key}` ({err})"));
                    }
                }
            }
        }
    }

    let targets = extract_spec_and_paper_targets(prose, embed_papers);
    if targets.is_empty() {
        issues.push(
            "verified `spec:<id>` (`docs/math_specs/<id>.md`) or `papers/<name>.tex` reference"
                .to_string(),
        );
    } else {
        for rel in targets {
            verify_target_file(workspace_root, bib_db, &rel, &mut issues);
        }
    }
    issues
}

fn verify_target_file(
    workspace_root: &Path,
    bib_db: &BibtexDatabase,
    rel: &str,
    issues: &mut Vec<String>,
) {
    let full = workspace_root.join(rel);
    if !full.is_file() {
        issues.push(format!("existing target `{rel}`"));
        return;
    }
    if let Some(stem) = rel
        .strip_prefix("docs/math_specs/")
        .and_then(|s| s.strip_suffix(".md"))
    {
        match fs::read_to_string(&full) {
            Ok(md) => {
                if let Err(err) = validate_math_spec_markdown(stem, &md, bib_db) {
                    issues.push(err);
                }
            }
            Err(err) => issues.push(format!("readable `{rel}` ({err})")),
        }
    }
}

fn extract_spec_and_paper_targets(prose: &str, embed_papers: &[String]) -> BTreeSet<String> {
    let mut targets: BTreeSet<String> = embed_papers.iter().cloned().collect();
    for spec_id in extract_doc_spec_ids(prose) {
        targets.insert(format!("docs/math_specs/{spec_id}.md"));
    }
    for token in prose.split(|c: char| {
        c.is_whitespace() || matches!(c, '`' | '(' | ')' | '[' | ']' | '"' | '\'' | ',' | ';')
    }) {
        let trimmed = token.trim_end_matches(['.', ':']);
        if let Some(rest) = trimmed.strip_prefix("papers/")
            && rest.ends_with(".tex")
            && !rest.contains("..")
        {
            targets.insert(trimmed.to_string());
        }
    }
    targets
}

fn has_stability_note(lower_prose: &str) -> bool {
    [
        "numerical stability",
        "stability:",
        "# stability",
        "error bound",
        "conditioning",
        "truncation error",
        "rounding error",
    ]
    .iter()
    .any(|kw| lower_prose.contains(kw))
}

fn has_validity_domain(lower_prose: &str) -> bool {
    [
        "domain of validity",
        "validity:",
        "# validity",
        "precondition",
        "valid for",
    ]
    .iter()
    .any(|kw| lower_prose.contains(kw))
}
