pub(crate) const MAX_ATTR_STACK_STEPS: usize = 256;
pub(crate) const MAX_TYPE_UNWRAP_STEPS: usize = 32;

/// Iteratively traverses `allow(...)`, `expect(...)`, and nested `cfg_attr(pred, ...)`
/// meta trees up to [`MAX_ATTR_STACK_STEPS`] stack steps, returning the first match
/// produced by `matcher` on a suppressed lint path.
pub(crate) fn find_suppressed_lint_in_meta<T, F>(root_meta: &syn::Meta, mut matcher: F) -> Option<T>
where
    F: FnMut(&syn::Path) -> Option<T>,
{
    let mut stack = vec![root_meta.clone()];
    let mut steps = 0usize;
    while let Some(meta) = stack.pop() {
        if steps >= MAX_ATTR_STACK_STEPS {
            break;
        }
        steps += 1;
        let syn::Meta::List(list) = meta else {
            continue;
        };
        if list.path.is_ident("allow") || list.path.is_ident("expect") {
            let Ok(nested) = list.parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            ) else {
                continue;
            };
            for item in nested {
                if let syn::Meta::Path(lint_path) = item
                    && let Some(found) = matcher(&lint_path)
                {
                    return Some(found);
                }
            }
        } else if list.path.is_ident("cfg_attr") {
            let Ok(nested) = list.parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            ) else {
                continue;
            };
            for item in nested.into_iter().skip(1) {
                stack.push(item);
            }
        }
    }
    None
}

/// Iteratively traverses `allow(...)`, `expect(...)`, and nested `cfg_attr(pred, ...)`
/// attributes up to [`MAX_ATTR_STACK_STEPS`] stack steps, returning the first match
/// produced by `matcher` on a suppressed lint path.
pub(crate) fn find_suppressed_lint_in_attr<T, F>(attr: &syn::Attribute, matcher: F) -> Option<T>
where
    F: FnMut(&syn::Path) -> Option<T>,
{
    find_suppressed_lint_in_meta(&attr.meta, matcher)
}

/// Unwraps `Group` and `Paren` wrappers around an `impl` self-type up to
/// [`MAX_TYPE_UNWRAP_STEPS`] steps and extracts the trailing segment identifier
/// when `qself` is absent.
pub(crate) fn impl_base_ident(ty: &syn::Type) -> Option<String> {
    let mut curr = ty;
    let mut depth = 0usize;
    while depth < MAX_TYPE_UNWRAP_STEPS {
        depth += 1;
        match curr {
            syn::Type::Group(g) => curr = &g.elem,
            syn::Type::Paren(p) => curr = &p.elem,
            syn::Type::Path(tp) if tp.qself.is_none() => {
                return tp.path.segments.last().map(|seg| seg.ident.to_string());
            }
            _ => return None,
        }
    }
    None
}

/// Formats `path:line` when `line > 0`, falling back to `path` when macro-generated
/// or synthetic spans report line `0`.
pub(crate) fn format_file_line(path: &str, line: usize) -> String {
    if line > 0 {
        format!("{path}:{line}")
    } else {
        path.to_string()
    }
}

/// Extracts the bound variable identifier from a local pattern up to
/// [`MAX_TYPE_UNWRAP_STEPS`] wrapper steps.
pub(crate) fn pat_ident_name(pat: &syn::Pat) -> Option<String> {
    let mut curr = pat;
    for _ in 0..MAX_TYPE_UNWRAP_STEPS {
        match curr {
            syn::Pat::Ident(pi) => return Some(pi.ident.to_string()),
            syn::Pat::Type(pt) => curr = &pt.pat,
            syn::Pat::Paren(pp) => curr = &pp.pat,
            syn::Pat::Reference(pr) => curr = &pr.pat,
            _ => return None,
        }
    }
    None
}
