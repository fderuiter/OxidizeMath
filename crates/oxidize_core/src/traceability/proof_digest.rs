//! Pure-`std` FIPS 180-4 SHA-256 digest and Lean 4 proof certificate verification helpers.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;

/// Typed error returned when reading or validating Lean 4 proof sources and SHA-256 digests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProofDigestError {
    /// A forbidden unproven keyword (`sorry`, `admit`, or `axiom`) appeared outside comments.
    ForbiddenKeyword {
        /// Lean 4 proof file name.
        filename: String,
        /// Matched forbidden keyword token.
        keyword: String,
    },
    /// Proof file contained zero `theorem` or `lemma` declarations outside comments.
    MissingTheoremDeclaration {
        /// Lean 4 proof file name.
        filename: String,
    },
    /// Reading the `proofs/OxidizeMath` directory failed.
    ReadProofDirectory {
        /// Directory path string.
        path: String,
        /// Underlying I/O error message.
        reason: String,
    },
    /// Reading an individual `.lean` proof file failed.
    ReadProofFile {
        /// File path string.
        path: String,
        /// Underlying I/O error message.
        reason: String,
    },
    /// No `.lean` proof files were found in `proofs/OxidizeMath`.
    EmptyProofDirectory {
        /// Directory path string.
        path: String,
    },
}

impl fmt::Display for ProofDigestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForbiddenKeyword { filename, keyword } => write!(
                f,
                "Forbidden unproven keyword `{keyword}` found in Lean 4 proof `{filename}`"
            ),
            Self::MissingTheoremDeclaration { filename } => write!(
                f,
                "Lean 4 proof `{filename}` contains no `theorem` or `lemma` declarations"
            ),
            Self::ReadProofDirectory { path, reason } => {
                write!(f, "Failed reading Lean 4 proof directory {path}: {reason}")
            }
            Self::ReadProofFile { path, reason } => write!(f, "Failed reading {path}: {reason}"),
            Self::EmptyProofDirectory { path } => {
                write!(f, "No `.lean` proof files found in {path}")
            }
        }
    }
}

impl std::error::Error for ProofDigestError {}

/// Canonical 6-tier scientific verification pyramid names recorded in AST scans and certificates.
pub const CANONICAL_PYRAMID_TIERS: [&str; 6] = [
    "Tier1_Convergence",
    "Tier2_Property",
    "Tier3_Golden",
    "Tier4_Stochastic",
    "Tier5_Interval",
    "Tier6_Formal",
];

/// Canonical count of mathematical solvers certified by the Method of Manufactured Solutions (Tier 1).
pub const CERTIFIED_SOLVER_COUNT: usize = 55;

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn compress_block(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (i, chunk) in block.as_chunks::<4>().0.iter().enumerate() {
        w[i] = u32::from_be_bytes(*chunk);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

/// Computes the deterministic lowercase 64-character hex FIPS 180-4 SHA-256 digest of `data`.
#[must_use]
pub fn sha256_hex(data: &[u8]) -> String {
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity((data.len() + 9).div_ceil(64) * 64);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for block in padded.as_chunks::<64>().0 {
        compress_block(&mut state, block);
    }

    let mut hex = String::with_capacity(64);
    for word in state {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{word:08x}");
    }
    hex
}

/// Strips Lean 4 line (`-- ...`) and nested block (`/- ... -/`) comments so keyword checks
/// inspect executable declarations only.
#[must_use]
pub fn strip_lean_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0usize;
    let mut block_depth = 0usize;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'-' {
            block_depth += 1;
            i += 2;
            continue;
        }
        if block_depth > 0 && i + 1 < bytes.len() && bytes[i] == b'-' && bytes[i + 1] == b'/' {
            block_depth -= 1;
            i += 2;
            out.push(' ');
            continue;
        }
        if block_depth == 0 && i + 1 < bytes.len() && bytes[i] == b'-' && bytes[i + 1] == b'-' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if block_depth == 0 {
            let ch = src[i..].chars().next().unwrap_or(' ');
            out.push(ch);
            i += ch.len_utf8();
        } else {
            i += 1;
        }
    }
    out
}

/// Validates that a Lean 4 source file contains at least one `theorem` or `lemma` and zero
/// `sorry`, `admit`, or `axiom` tokens outside comments.
pub fn validate_lean_proof_source(filename: &str, src: &str) -> Result<(), ProofDigestError> {
    let stripped = strip_lean_comments(src);
    let mut has_theorem = false;
    for tok in stripped.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
        match tok {
            "sorry" | "admit" | "axiom" => {
                return Err(ProofDigestError::ForbiddenKeyword {
                    filename: filename.to_string(),
                    keyword: tok.to_string(),
                });
            }
            "theorem" | "lemma" => has_theorem = true,
            _ => {}
        }
    }
    if !has_theorem {
        return Err(ProofDigestError::MissingTheoremDeclaration {
            filename: filename.to_string(),
        });
    }
    Ok(())
}

/// Scans `proofs/OxidizeMath/*.lean` under `workspace_root`, validates every proof file is free
/// of `sorry`/`admit`/`axiom`, and returns deterministic SHA-256 digests keyed by file name.
pub fn collect_lean4_proof_digests(
    workspace_root: &Path,
) -> Result<BTreeMap<String, String>, ProofDigestError> {
    let dir = workspace_root.join("proofs").join("OxidizeMath");
    let entries = fs::read_dir(&dir).map_err(|err| ProofDigestError::ReadProofDirectory {
        path: dir.display().to_string(),
        reason: err.to_string(),
    })?;
    let mut digests = BTreeMap::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("lean") {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let normalized = fs::read_to_string(&path)
            .map_err(|err| ProofDigestError::ReadProofFile {
                path: path.display().to_string(),
                reason: err.to_string(),
            })?
            .replace("\r\n", "\n");
        validate_lean_proof_source(name, &normalized)?;
        digests.insert(name.to_string(), sha256_hex(normalized.as_bytes()));
    }
    if digests.is_empty() {
        return Err(ProofDigestError::EmptyProofDirectory {
            path: dir.display().to_string(),
        });
    }
    Ok(digests)
}

#[cfg(test)]
mod tests {
    use super::{sha256_hex, strip_lean_comments, validate_lean_proof_source};

    #[test]
    fn sha256_matches_nist_fips_180_4_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn lean_proof_validator_accepts_complete_theorems_and_rejects_sorry() {
        let valid = "/- sorry mentioned in comment -/\ntheorem demo (x : Int) : x = x := rfl\n";
        assert!(validate_lean_proof_source("Demo.lean", valid).is_ok());
        assert!(!strip_lean_comments(valid).contains("sorry"));

        let broken = "theorem bad (x : Int) : x = x + 1 := by sorry\n";
        assert!(validate_lean_proof_source("Bad.lean", broken).is_err());
    }
}
