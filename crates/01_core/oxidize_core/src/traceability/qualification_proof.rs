//! Lean 4 `Tier6_Formal` proof and theorem binding validation for `Certified` maturity targets.

use crate::ast_visitor::FormalProofMacroBinding;
use crate::traceability::proof_digest::{extract_lean_theorem_names, validate_lean_proof_source};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

/// Workspace Lean 4 proof catalog used to validate `formal_proof_verification!` bindings.
#[derive(Clone, Debug, Default)]
pub(super) struct WorkspaceProofCatalog {
    /// Whether `proofs/OxidizeMath` exists in `workspace_root`.
    pub has_proofs_dir: bool,
    /// Whether `proofs/lakefile.lean` exists in `workspace_root` (requiring compiled `.olean` outputs).
    pub has_lakefile: bool,
    /// Exported public theorem names keyed by `.lean` proof filename.
    pub declared_theorems: BTreeMap<String, BTreeSet<String>>,
    /// Proof filenames with compiled `.olean` artifacts under `proofs/.lake/build/lib/OxidizeMath/`.
    pub compiled_proofs: BTreeSet<String>,
    /// Source-level hygiene errors per `.lean` proof filename.
    pub proof_errors: BTreeMap<String, String>,
}

impl WorkspaceProofCatalog {
    pub(super) fn load(workspace_root: &Path) -> Self {
        let proofs_root = workspace_root.join("proofs");
        let ox_dir = proofs_root.join("OxidizeMath");
        let olean_dir = proofs_root
            .join(".lake")
            .join("build")
            .join("lib")
            .join("OxidizeMath");
        let has_lakefile = proofs_root.join("lakefile.lean").is_file();
        let Ok(entries) = fs::read_dir(&ox_dir) else {
            return Self {
                has_proofs_dir: false,
                has_lakefile,
                ..Self::default()
            };
        };

        let mut catalog = Self {
            has_proofs_dir: true,
            has_lakefile,
            ..Self::default()
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("lean") {
                continue;
            }
            let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            match fs::read_to_string(&path) {
                Ok(raw) => {
                    let normalized = raw.replace("\r\n", "\n");
                    if let Err(err) = validate_lean_proof_source(filename, &normalized) {
                        catalog
                            .proof_errors
                            .insert(filename.to_string(), err.to_string());
                    } else {
                        let thms: BTreeSet<String> = extract_lean_theorem_names(&normalized)
                            .into_iter()
                            .collect();
                        catalog.declared_theorems.insert(filename.to_string(), thms);
                    }
                }
                Err(err) => {
                    catalog
                        .proof_errors
                        .insert(filename.to_string(), err.to_string());
                }
            }
            if olean_dir.join(format!("{stem}.olean")).is_file() {
                catalog.compiled_proofs.insert(filename.to_string());
            }
        }
        catalog
    }

    fn validate_binding_syntax(
        rel_path: &str,
        binding: &FormalProofMacroBinding,
        violations: &mut Vec<String>,
    ) -> Option<String> {
        let loc = format!("{rel_path}:{}", binding.line);
        let proof = binding.proof.trim();
        if proof.is_empty()
            || !proof.ends_with(".lean")
            || proof.contains('/')
            || proof.contains('\\')
            || proof.contains("..")
        {
            violations.push(format!(
                "{loc}: `formal_proof_verification!` for module `{}` has invalid `proof = \"{}\"` (expected `<Module>.lean` in `proofs/OxidizeMath/`).",
                binding.module, binding.proof
            ));
            return None;
        }
        if !binding.has_assertions {
            violations.push(format!(
                "{loc}: `formal_proof_verification!` for module `{}` (`{proof}`) has vacuous `test = {{}}` without assertions.",
                binding.module
            ));
            return None;
        }
        if binding.theorems.is_empty() || binding.theorems.iter().any(|t| t.trim().is_empty()) {
            violations.push(format!(
                "{loc}: `formal_proof_verification!` for module `{}` (`{proof}`) is missing non-empty `theorems = [\"...\"]` bindings.",
                binding.module
            ));
            return None;
        }
        Some(proof.to_string())
    }

    pub(super) fn validate_binding(
        &self,
        rel_path: &str,
        binding: &FormalProofMacroBinding,
        violations: &mut Vec<String>,
    ) -> bool {
        let Some(proof_owned) = Self::validate_binding_syntax(rel_path, binding, violations) else {
            return false;
        };
        let proof = proof_owned.as_str();
        let loc = format!("{rel_path}:{}", binding.line);
        if !self.has_proofs_dir {
            violations.push(format!(
                "{loc}: `formal_proof_verification!` for module `{}` references `{proof}`, but `proofs/OxidizeMath/` does not exist.",
                binding.module
            ));
            return false;
        }
        if let Some(err) = self.proof_errors.get(proof) {
            violations.push(format!(
                "{loc}: `formal_proof_verification!` for module `{}` references invalid Lean proof `{proof}`: {err}",
                binding.module
            ));
            return false;
        }
        let Some(declared) = self.declared_theorems.get(proof) else {
            violations.push(format!(
                "{loc}: `formal_proof_verification!` for module `{}` references missing Lean proof `proofs/OxidizeMath/{proof}`.",
                binding.module
            ));
            return false;
        };
        if self.has_lakefile && !self.compiled_proofs.contains(proof) {
            violations.push(format!(
                "{loc}: Lean proof `proofs/OxidizeMath/{proof}` bound to `{}` has no compiled `.olean` artifact; run `lake build` in `proofs/`.",
                binding.module
            ));
            return false;
        }
        let mut all_found = true;
        for thm in &binding.theorems {
            if !declared.contains(thm) {
                violations.push(format!(
                    "{loc}: `formal_proof_verification!` for module `{}` binds unknown theorem `{thm}` not declared as a public theorem in `{proof}`.",
                    binding.module
                ));
                all_found = false;
            }
        }
        all_found
    }
}
