use oxidize_core::traceability::LeanProofBinding;
use std::collections::BTreeMap;
use std::path::Path;

/// Complete Lean 4 `Tier6_Formal` verification result combining SHA-256 digests and kernel bindings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeanVerificationSummary {
    /// Deterministic FIPS 180-4 SHA-256 digests keyed by `.lean` proof filename.
    pub digests: BTreeMap<String, String>,
    /// Kernel-verified status, bound Rust modules, and theorem names per `.lean` proof filename.
    pub bindings: BTreeMap<String, LeanProofBinding>,
}

#[cfg(not(target_arch = "wasm32"))]
mod host {
    use super::{LeanProofBinding, LeanVerificationSummary};
    use oxidize_core::ast_visitor::AstVisitor;
    use oxidize_core::traceability::{collect_lean4_proof_digests, discover_maturity_scan_files};
    use std::collections::{BTreeMap, BTreeSet};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};

    pub(super) fn resolve_workspace_root() -> PathBuf {
        if let Some(root) = oxidize_core::workspace::find_workspace_root()
            && root.join("proofs").join("lakefile.lean").is_file()
        {
            return root;
        }
        if let Some(manifest_root) =
            oxidize_core::workspace::find_workspace_root_from(Path::new(env!("CARGO_MANIFEST_DIR")))
            && manifest_root.join("proofs").join("lakefile.lean").is_file()
        {
            return manifest_root;
        }
        PathBuf::from(".")
    }

    const KERNEL_INSPECTOR_BODY: &str = r#"
open Lean

def allowedAxiom (n : Name) : Bool :=
  n == ``propext || n == ``Quot.sound || n == ``Classical.choice

def isCompilerStage (n : Name) : Bool :=
  match n with
  | .str _ s => s.startsWith "_cstage" || s.startsWith "_closed_" || s.startsWith "_regBuiltin"
  | _ => false

def oxidizeModuleStem? (env : Environment) (n : Name) : Option String := do
  let idx ← env.getModuleIdxFor? n
  let modName ← env.header.moduleNames.get? idx.toNat
  match modName with
  | .str (.str .anonymous "OxidizeMath") stem => some stem
  | .str .anonymous "OxidizeMath" => some "OxidizeMath"
  | _ => none

def inspectEnv : MetaM Unit := do
  let env ← getEnv
  let mut violations : Array String := #[]
  let mut theorems : Array String := #[]
  for (name, cinfo) in env.constants.toList do
    if let some stem := oxidizeModuleStem? env name then
      let modFile := s!"{stem}.lean"
      if Lean.isExtern env name then
        violations := violations.push s!"{modFile}: @[extern] attribute banned on `{name}`"
      if (Lean.Compiler.getImplementedBy? env name).isSome then
        violations := violations.push s!"{modFile}: @[implemented_by] attribute banned on `{name}`"
      match cinfo with
      | .axiomInfo _ =>
        violations := violations.push s!"{modFile}: user `axiom` declaration banned on `{name}`"
      | .opaqueInfo _ =>
        if !isCompilerStage name then
          violations := violations.push s!"{modFile}: `opaque` declaration banned on `{name}`"
      | .thmInfo _ =>
        let axs ← Lean.collectAxioms name
        for ax in axs do
          if !allowedAxiom ax then
            violations := violations.push s!"{modFile}: banned axiom `{ax}` used by `{name}`"
        if !name.isInternal then
          if let .str parent shortName := name then
            if parent == .str (.str .anonymous "OxidizeMath") stem then
              theorems := theorems.push s!"{modFile}:{shortName}"
      | .defnInfo _ =>
        if !isCompilerStage name then
          let axs ← Lean.collectAxioms name
          for ax in axs do
            if !allowedAxiom ax then
              violations := violations.push s!"{modFile}: banned axiom `{ax}` used by `{name}`"
      | _ => pure ()
  if !violations.isEmpty then
    for v in violations do
      IO.eprintln s!"LEAN_VIOLATION:{v}"
    throwError "Lean kernel axiom/attribute hygiene failed"
  for t in theorems.qsort (· < ·) do
    IO.println s!"VERIFIED_THEOREM:{t}"

#eval inspectEnv
"#;

    fn build_kernel_inspector_script(proof_files: &BTreeMap<String, String>) -> String {
        let mut script = String::from("import Lean\n");
        for filename in proof_files.keys() {
            if let Some(stem) = filename.strip_suffix(".lean") {
                script.push_str(&format!("import OxidizeMath.{stem}\n"));
            }
        }
        script.push_str(KERNEL_INSPECTOR_BODY);
        script
    }

    fn run_lake_build(proofs_dir: &Path, lake_bin: &str) -> Result<(), String> {
        let probe = Command::new(lake_bin)
        .arg("--version")
        .current_dir(proofs_dir)
        .output()
        .map_err(|err| {
            format!(
                "Lean 4 `lake` binary `{lake_bin}` is required for Tier6_Formal verification but could not be executed ({err}). Run `cargo run -p xtask -- setup --install` or install `elan`."
            )
        })?;
        if !probe.status.success() {
            return Err(format!(
                "`{lake_bin} --version` failed in {}:\n{}{}",
                proofs_dir.display(),
                String::from_utf8_lossy(&probe.stdout),
                String::from_utf8_lossy(&probe.stderr)
            ));
        }

        let build = Command::new(lake_bin)
            .arg("build")
            .current_dir(proofs_dir)
            .output()
            .map_err(|err| {
                format!(
                    "Failed spawning `{lake_bin} build` in {}: {err}",
                    proofs_dir.display()
                )
            })?;
        if !build.status.success() {
            return Err(format!(
                "`{lake_bin} build` failed in {}:\n{}{}",
                proofs_dir.display(),
                String::from_utf8_lossy(&build.stdout),
                String::from_utf8_lossy(&build.stderr)
            ));
        }
        Ok(())
    }

    fn inspect_compiled_lean_environment(
        proofs_dir: &Path,
        lake_bin: &str,
        digests: &BTreeMap<String, String>,
    ) -> Result<BTreeMap<String, BTreeSet<String>>, String> {
        let script = build_kernel_inspector_script(digests);
        let mut child = Command::new(lake_bin)
            .args(["env", "lean", "--stdin"])
            .current_dir(proofs_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|err| {
                format!(
                    "Failed spawning `{lake_bin} env lean --stdin` in {}: {err}",
                    proofs_dir.display()
                )
            })?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(script.as_bytes())
                .map_err(|err| format!("Failed writing Lean kernel inspector script: {err}"))?;
        }
        let output = child
            .wait_with_output()
            .map_err(|err| format!("Failed waiting for Lean kernel inspector: {err}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !output.status.success() {
            return Err(format!(
                "Lean 4 kernel axiom/attribute inspection failed in {}:\n{stdout}{stderr}",
                proofs_dir.display()
            ));
        }

        let mut verified_theorems: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for line in stdout.lines() {
            if let Some(rest) = line.trim().strip_prefix("VERIFIED_THEOREM:")
                && let Some((mod_file, thm)) = rest.split_once(':')
            {
                verified_theorems
                    .entry(mod_file.to_string())
                    .or_default()
                    .insert(thm.to_string());
            }
        }

        for filename in digests.keys() {
            if verified_theorems
                .get(filename)
                .is_none_or(BTreeSet::is_empty)
            {
                return Err(format!(
                    "Lean 4 proof `{filename}` exported zero kernel-verified theorems in namespace `OxidizeMath`."
                ));
            }
        }
        Ok(verified_theorems)
    }

    fn verify_file_formal_bindings(
        rel: &str,
        src: &str,
        kernel_theorems: &BTreeMap<String, BTreeSet<String>>,
        modules_by_proof: &mut BTreeMap<String, BTreeSet<String>>,
        theorems_by_proof: &mut BTreeMap<String, BTreeSet<String>>,
    ) -> Result<(), String> {
        let Ok(ast) = syn::parse_file(src) else {
            return Ok(());
        };
        let mut visitor = AstVisitor::new();
        syn::visit::Visit::visit_file(&mut visitor, &ast);
        for fb in visitor.formal_proof_bindings {
            let Some(compiled_thms) = kernel_theorems.get(&fb.proof) else {
                return Err(format!(
                    "{rel}:{}: `formal_proof_verification!` for `{}` references unknown Lean proof `{}`",
                    fb.line, fb.module, fb.proof
                ));
            };
            if fb.theorems.is_empty() {
                return Err(format!(
                    "{rel}:{}: `formal_proof_verification!` for `{}` (`{}`) must declare non-empty `theorems = [...]`",
                    fb.line, fb.module, fb.proof
                ));
            }
            for thm in fb.theorems {
                if !compiled_thms.contains(&thm) {
                    return Err(format!(
                        "{rel}:{}: theorem `{thm}` bound by `{}` was not found in compiled Lean kernel environment for `{}`",
                        fb.line, fb.module, fb.proof
                    ));
                }
                theorems_by_proof
                    .entry(fb.proof.clone())
                    .or_default()
                    .insert(thm);
            }
            modules_by_proof
                .entry(fb.proof)
                .or_default()
                .insert(fb.module);
        }
        Ok(())
    }

    fn collect_and_verify_rust_bindings(
        workspace_root: &Path,
        digests: &BTreeMap<String, String>,
        kernel_theorems: &BTreeMap<String, BTreeSet<String>>,
    ) -> Result<BTreeMap<String, LeanProofBinding>, String> {
        let mut modules_by_proof: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut theorems_by_proof: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

        for path in discover_maturity_scan_files(workspace_root) {
            let rel = path
                .strip_prefix(workspace_root)
                .unwrap_or(path.as_path())
                .to_string_lossy()
                .replace('\\', "/");
            if rel.contains("/tests/") {
                continue;
            }
            let Ok(src) = std::fs::read_to_string(&path) else {
                continue;
            };
            if !src.contains("formal_proof_verification") {
                continue;
            }
            verify_file_formal_bindings(
                &rel,
                &src,
                kernel_theorems,
                &mut modules_by_proof,
                &mut theorems_by_proof,
            )?;
        }

        let mut bindings = BTreeMap::new();
        for proof_file in digests.keys() {
            let rust_modules: Vec<String> = modules_by_proof
                .remove(proof_file)
                .unwrap_or_default()
                .into_iter()
                .collect();
            let theorems: Vec<String> = theorems_by_proof
                .remove(proof_file)
                .or_else(|| kernel_theorems.get(proof_file).cloned())
                .unwrap_or_default()
                .into_iter()
                .collect();
            if rust_modules.is_empty() {
                return Err(format!(
                    "Lean 4 proof `{proof_file}` has no Rust `formal_proof_verification!` module binding."
                ));
            }
            bindings.insert(
                proof_file.clone(),
                LeanProofBinding {
                    kernel_verified: true,
                    rust_modules,
                    theorems,
                },
            );
        }
        Ok(bindings)
    }

    pub(super) fn verify_in_workspace(
        workspace_root: &Path,
        lake_bin: &str,
        min_proof_files: usize,
    ) -> Result<LeanVerificationSummary, String> {
        static LAKE_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LAKE_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let proofs_dir = workspace_root.join("proofs");
        if !proofs_dir.join("lakefile.lean").is_file()
            || !proofs_dir.join("lean-toolchain").is_file()
        {
            return Err(format!(
                "Missing Lean 4 Lake configuration in {}",
                proofs_dir.display()
            ));
        }
        let digests = collect_lean4_proof_digests(workspace_root).map_err(|err| err.to_string())?;
        if digests.len() < min_proof_files {
            return Err(format!(
                "Expected at least {min_proof_files} Lean 4 proof modules in proofs/OxidizeMath/, found {}",
                digests.len()
            ));
        }
        run_lake_build(&proofs_dir, lake_bin)?;
        let kernel_theorems = inspect_compiled_lean_environment(&proofs_dir, lake_bin, &digests)?;
        let bindings =
            collect_and_verify_rust_bindings(workspace_root, &digests, &kernel_theorems)?;
        Ok(LeanVerificationSummary { digests, bindings })
    }
}

/// Runs mandatory `lake build`, Lean 4 kernel axiom/attribute inspection, and Rust theorem-binding
/// cross-checks against `workspace_root` using `lake_bin` (`"lake"` in production).
pub fn verify_lean_proofs_in_workspace(
    workspace_root: &Path,
    lake_bin: &str,
    min_proof_files: usize,
) -> Result<LeanVerificationSummary, String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        host::verify_in_workspace(workspace_root, lake_bin, min_proof_files)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (workspace_root, lake_bin, min_proof_files);
        Err("Tier6_Formal Lean 4 kernel verification requires a native host toolchain".to_string())
    }
}

/// Verifies all committed workspace Lean 4 proofs via mandatory `lake build` and kernel axiom inspection.
pub fn verify_lean_proofs() -> Result<LeanVerificationSummary, String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let root = host::resolve_workspace_root();
        verify_lean_proofs_in_workspace(&root, "lake", 6)
    }
    #[cfg(target_arch = "wasm32")]
    {
        Err("Tier6_Formal Lean 4 kernel verification requires a native host toolchain".to_string())
    }
}

/// Verifies all committed Lean 4 proofs via `lake build` and kernel axiom inspection, returning SHA-256 digests.
pub fn verify_and_digest_lean_proofs() -> Result<BTreeMap<String, String>, String> {
    Ok(verify_lean_proofs()?.digests)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "lean_proofs_tests.rs"]
mod tests;
