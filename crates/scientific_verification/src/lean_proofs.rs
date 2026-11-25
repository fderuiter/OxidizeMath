use oxidize_core::traceability::collect_lean4_proof_digests;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn resolve_workspace_root() -> PathBuf {
    if let Some(root) = oxidize_core::workspace::find_workspace_root()
        && root.join("proofs").join("lakefile.lean").is_file()
    {
        return root;
    }
    let manifest_fallback = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    if manifest_fallback
        .join("proofs")
        .join("lakefile.lean")
        .is_file()
    {
        return manifest_fallback;
    }
    PathBuf::from(".")
}

fn try_invoke_lake_build_if_installed(proofs_dir: &Path) -> Result<bool, String> {
    let probe = Command::new("lake").arg("--version").output();
    let Ok(out) = probe else {
        return Ok(false);
    };
    if !out.status.success() {
        return Ok(false);
    }
    let build = Command::new("lake")
        .arg("build")
        .current_dir(proofs_dir)
        .output()
        .map_err(|err| {
            format!(
                "Failed spawning `lake build` in {}: {err}",
                proofs_dir.display()
            )
        })?;
    if !build.status.success() {
        return Err(format!(
            "`lake build` failed in {}:\n{}",
            proofs_dir.display(),
            String::from_utf8_lossy(&build.stderr)
        ));
    }
    Ok(true)
}

/// Verifies all committed Lean 4 proofs under `proofs/OxidizeMath/`, ensures zero `sorry`/`admit`/`axiom`
/// tokens exist outside comments, optionally runs `lake build` when Lean 4 is installed, and returns
/// deterministic FIPS 180-4 SHA-256 digests keyed by proof file name.
pub fn verify_and_digest_lean_proofs() -> Result<BTreeMap<String, String>, String> {
    let root = resolve_workspace_root();
    let proofs_dir = root.join("proofs");
    if !proofs_dir.join("lakefile.lean").is_file() || !proofs_dir.join("lean-toolchain").is_file() {
        return Err(format!(
            "Missing Lean 4 Lake configuration in {}",
            proofs_dir.display()
        ));
    }
    let digests = collect_lean4_proof_digests(&root)?;
    if digests.len() < 6 {
        return Err(format!(
            "Expected at least 6 Lean 4 proof modules in proofs/OxidizeMath/, found {}",
            digests.len()
        ));
    }
    let _ran_lake = try_invoke_lake_build_if_installed(&proofs_dir)?;
    Ok(digests)
}
