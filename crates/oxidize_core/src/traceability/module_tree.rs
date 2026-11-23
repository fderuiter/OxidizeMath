//! Standard Rust module tree and `#[path = "..."]` submodule resolution helpers.

pub(super) struct ExternalSubmodule {
    pub ident: String,
    pub custom_path: Option<String>,
}

#[derive(Default)]
struct SubmoduleCollector {
    submodules: Vec<ExternalSubmodule>,
}

fn extract_path_attr(attrs: &[syn::Attribute]) -> Option<String> {
    for attr in attrs {
        if attr.path().is_ident("path")
            && let syn::Meta::NameValue(nv) = &attr.meta
            && let syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(s),
                ..
            }) = &nv.value
        {
            return Some(s.value());
        }
    }
    None
}

impl<'ast> syn::visit::Visit<'ast> for SubmoduleCollector {
    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        if node.content.is_none() {
            self.submodules.push(ExternalSubmodule {
                ident: node.ident.to_string(),
                custom_path: extract_path_attr(&node.attrs),
            });
        }
        syn::visit::visit_item_mod(self, node);
    }
}

pub(super) fn collect_external_submodules(ast: Option<&syn::File>) -> Vec<ExternalSubmodule> {
    let Some(ast) = ast else {
        return Vec::new();
    };
    let mut collector = SubmoduleCollector::default();
    syn::visit::Visit::visit_file(&mut collector, ast);
    collector.submodules
}

pub(super) fn module_parent_dir(norm_file_path: &str) -> String {
    norm_file_path
        .rsplit_once('/')
        .map_or_else(String::new, |(parent, _)| parent.to_string())
}

pub(super) fn module_child_dir(norm_file_path: &str) -> String {
    let mut dir_parts: Vec<&str> = norm_file_path.split('/').collect();
    let is_mod_rs = norm_file_path.ends_with("mod.rs")
        || norm_file_path.ends_with("lib.rs")
        || norm_file_path.ends_with("main.rs");
    if is_mod_rs {
        dir_parts.pop();
    } else if let Some(stem) = dir_parts.pop().map(|s| s.strip_suffix(".rs").unwrap_or(s)) {
        dir_parts.push(stem);
    }
    dir_parts.join("/")
}

impl<V: crate::vfs::VirtualFileSystem> super::TraceabilityEngine<V> {
    pub(super) async fn gather_active_files(
        &self,
        code_dirs: &[&str],
        active_files: &mut std::collections::HashSet<String>,
        cache: &mut super::AstCache,
    ) -> Result<(), std::io::Error> {
        for dir in code_dirs {
            let normalized = crate::path_utils::normalize_path(dir);
            let mut has_entrypoint = false;
            for entry_name in ["lib.rs", "main.rs", "mod.rs"] {
                let entry = crate::path_utils::join_and_normalize(&normalized, entry_name);
                if self.load_cached_optional(&entry, cache).await? {
                    self.parse_module_tree(&entry, active_files, cache).await?;
                    has_entrypoint = true;
                }
            }
            if !has_entrypoint {
                self.gather_files_fallback(&normalized, active_files)
                    .await?;
            }
        }
        Ok(())
    }

    async fn gather_files_fallback(
        &self,
        start_dir: &str,
        active_files: &mut std::collections::HashSet<String>,
    ) -> Result<(), std::io::Error> {
        let mut stack = vec![start_dir.to_string()];
        while let Some(current_dir) = stack.pop() {
            let entries = match self.vfs.list_dir(&current_dir).await {
                Ok(entries) => entries,
                Err(err)
                    if matches!(
                        err.kind(),
                        std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
                    ) =>
                {
                    continue;
                }
                Err(err) => return Err(err),
            };
            for name in entries {
                let path = crate::path_utils::join_and_normalize(&current_dir, &name);
                if name.ends_with(".rs") {
                    active_files.insert(path);
                } else if !name.contains('.') {
                    stack.push(path);
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::double_must_use)]
    #[async_recursion::async_recursion(?Send)]
    async fn parse_module_tree(
        &self,
        file_path: &str,
        active_files: &mut std::collections::HashSet<String>,
        cache: &mut super::AstCache,
    ) -> Result<(), std::io::Error> {
        let norm_file_path = crate::path_utils::normalize_path(file_path);
        if active_files.contains(&norm_file_path) {
            return Ok(());
        }
        let key = super::AstCache::normalize_key(&norm_file_path);
        if cache.get(&key).is_none() {
            let content = self.vfs.read_to_string(&norm_file_path).await?;
            cache.insert_source(key.clone(), content);
        }
        active_files.insert(norm_file_path.clone());

        let submodules = collect_external_submodules(cache.get(&key).and_then(|e| e.ast.as_ref()));
        let parent_dir = module_parent_dir(&norm_file_path);
        let dir_path = module_child_dir(&norm_file_path);
        for submodule in submodules {
            if let Some(rel_path) = submodule.custom_path {
                let custom = crate::path_utils::join_and_normalize(
                    &parent_dir,
                    rel_path.trim_start_matches("./"),
                );
                if self.load_cached_optional(&custom, cache).await? {
                    self.parse_module_tree(&custom, active_files, cache).await?;
                }
                continue;
            }
            let path1 =
                crate::path_utils::join_and_normalize(&dir_path, format!("{}.rs", submodule.ident));
            let path2 = crate::path_utils::join_and_normalize(
                &dir_path,
                format!("{}/mod.rs", submodule.ident),
            );

            if self.load_cached_optional(&path1, cache).await? {
                self.parse_module_tree(&path1, active_files, cache).await?;
            } else if self.load_cached_optional(&path2, cache).await? {
                self.parse_module_tree(&path2, active_files, cache).await?;
            }
        }
        Ok(())
    }
}
