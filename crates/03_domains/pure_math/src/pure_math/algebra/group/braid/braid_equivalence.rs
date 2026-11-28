//! Dehornoy handle reduction, modular Burau hashing, mixed-sign Yang-Baxter rewriting,
//! and validated deserialization for [`BraidWord`].

// [cite:knot_braid_topology]

use super::{BraidWord, MAX_BRAID_STRANDS};
use math_commons::math_kernel::types::flatten_2d_index;
use std::hash::{Hash, Hasher};

const MOD_P1: u64 = 1_000_000_007;
const MOD_T1: u64 = 257;
const MOD_T1_INV: u64 = mod_pow_u64(MOD_T1, MOD_P1 - 2, MOD_P1);

const MOD_P2: u64 = 1_000_000_009;
const MOD_T2: u64 = 521;
const MOD_T2_INV: u64 = mod_pow_u64(MOD_T2, MOD_P2 - 2, MOD_P2);

const _: () = assert!((MOD_T1 * MOD_T1_INV) % MOD_P1 == 1);
const _: () = assert!((MOD_T2 * MOD_T2_INV) % MOD_P2 == 1);

const MAX_DEHORNOY_STEPS: usize = 65_536;
const MAX_DEHORNOY_BUFFER: usize = 32_768;

const fn mod_pow_u64(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut acc = 1_u64;
    base %= modulus;
    while exp > 0 {
        if (exp & 1) == 1 {
            acc = ((acc as u128 * base as u128) % modulus as u128) as u64;
        }
        exp >>= 1;
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
    }
    acc
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBraidWord {
    strands: usize,
    generators: Vec<i32>,
}

impl<'de> serde::Deserialize<'de> for BraidWord {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawBraidWord::deserialize(deserializer)?;
        Self::new(raw.strands, raw.generators).map_err(serde::de::Error::custom)
    }
}

#[verified_engine::verified]
pub(super) fn match_artin_triple(g0: i32, g1: i32, g2: i32) -> Option<[i32; 3]> {
    let i = g0.unsigned_abs();
    let j = g1.unsigned_abs();
    if i == 0 || j == 0 || i.abs_diff(j) != 1 {
        return None;
    }
    if g0 == g2 && g0.signum() == g1.signum() {
        return Some([g1, g0, g1]);
    }
    if g2 == -g0 {
        let i_signed = i as i32;
        let j_signed = j as i32;
        let s0 = g0.signum();
        let s1 = g1.signum();
        return Some([-s0 * j_signed, s1 * i_signed, s0 * j_signed]);
    }
    None
}

impl PartialEq for BraidWord {
    #[verified_engine::verified]
    fn eq(&self, other: &Self) -> bool {
        if self.strands != other.strands || self.writhe() != other.writhe() {
            return false;
        }
        if self.generators == other.generators {
            return true;
        }
        let dim = self.strands.saturating_sub(1);
        if dim > 0 {
            let m1_a = eval_modular_burau(dim, &self.generators, MOD_P1, MOD_T1, MOD_T1_INV);
            let m1_b = eval_modular_burau(dim, &other.generators, MOD_P1, MOD_T1, MOD_T1_INV);
            if m1_a != m1_b {
                return false;
            }
            let m2_a = eval_modular_burau(dim, &self.generators, MOD_P2, MOD_T2, MOD_T2_INV);
            let m2_b = eval_modular_burau(dim, &other.generators, MOD_P2, MOD_T2, MOD_T2_INV);
            if m2_a != m2_b {
                return false;
            }
        }
        let mut diff = Vec::with_capacity(self.generators.len() + other.generators.len());
        for &g in &self.generators {
            push_free_reduced(&mut diff, g);
        }
        for &g in other.generators.iter().rev() {
            push_free_reduced(&mut diff, -g);
        }
        dehornoy_reduces_to_identity(diff)
    }
}

impl Eq for BraidWord {}

impl Hash for BraidWord {
    #[verified_engine::verified]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.strands.hash(state);
        self.writhe().hash(state);
        let dim = self.strands.saturating_sub(1);
        if dim == 0 {
            return;
        }
        let m1 = eval_modular_burau(dim, &self.generators, MOD_P1, MOD_T1, MOD_T1_INV);
        let m2 = eval_modular_burau(dim, &self.generators, MOD_P2, MOD_T2, MOD_T2_INV);
        m1[..(dim * dim)].hash(state);
        m2[..(dim * dim)].hash(state);
    }
}

#[inline]
#[verified_engine::verified]
pub(super) fn push_free_reduced(stack: &mut Vec<i32>, g: i32) {
    if stack.last().is_some_and(|&top| top == -g) {
        stack.pop();
    } else {
        stack.push(g);
    }
}

#[verified_engine::verified]
fn eval_modular_burau(
    dim: usize,
    generators: &[i32],
    p: u64,
    t: u64,
    inv_t: u64,
) -> [u64; (MAX_BRAID_STRANDS - 1) * (MAX_BRAID_STRANDS - 1)] {
    let mut mat = [0_u64; (MAX_BRAID_STRANDS - 1) * (MAX_BRAID_STRANDS - 1)];
    for i in 0..dim {
        mat[flatten_2d_index(i, i, dim)] = 1;
    }
    let neg_t = (p - t) % p;
    let neg_inv_t = (p - inv_t) % p;

    for &g in generators {
        let k = (g.unsigned_abs() as usize) - 1;
        for r in 0..dim {
            let rk = flatten_2d_index(r, k, dim);
            let m_rk = mat[rk];
            if m_rk == 0 {
                continue;
            }
            if g > 0 {
                if k > 0 {
                    let prev = flatten_2d_index(r, k - 1, dim);
                    mat[prev] = (mat[prev] + (t * m_rk) % p) % p;
                }
                mat[rk] = (neg_t * m_rk) % p;
                if k + 1 < dim {
                    let next = flatten_2d_index(r, k + 1, dim);
                    mat[next] = (mat[next] + m_rk) % p;
                }
            } else {
                if k > 0 {
                    let prev = flatten_2d_index(r, k - 1, dim);
                    mat[prev] = (mat[prev] + m_rk) % p;
                }
                mat[rk] = (neg_inv_t * m_rk) % p;
                if k + 1 < dim {
                    let next = flatten_2d_index(r, k + 1, dim);
                    mat[next] = (mat[next] + (inv_t * m_rk) % p) % p;
                }
            }
        }
    }
    mat
}

#[verified_engine::verified]
fn dehornoy_reduces_to_identity(mut word: Vec<i32>) -> bool {
    for _ in 0..MAX_DEHORNOY_STEPS {
        if word.is_empty() {
            return true;
        }
        let Some((p, q)) = find_first_finishing_handle(&word) else {
            return false;
        };
        if word.len() > MAX_DEHORNOY_BUFFER {
            return false;
        }
        word = rewrite_dehornoy_handle(&word, p, q);
    }
    word.is_empty()
}

#[verified_engine::verified]
fn find_first_finishing_handle(word: &[i32]) -> Option<(usize, usize)> {
    for q in 1..word.len() {
        let g_q = word[q];
        let idx_q = g_q.unsigned_abs();
        for p in (0..q).rev() {
            let g_p = word[p];
            let idx_p = g_p.unsigned_abs();
            if idx_p < idx_q {
                break;
            }
            if idx_p == idx_q {
                if g_p == -g_q {
                    return Some((p, q));
                }
                break;
            }
        }
    }
    None
}

#[verified_engine::verified]
fn rewrite_dehornoy_handle(word: &[i32], p: usize, q: usize) -> Vec<i32> {
    let g_p = word[p];
    let i = g_p.unsigned_abs() as i32;
    let eps = g_p.signum();
    let mut out = Vec::with_capacity(word.len() + 4);
    for &g in &word[..p] {
        push_free_reduced(&mut out, g);
    }
    for &g in &word[(p + 1)..q] {
        if g.unsigned_abs() as i32 == i + 1 {
            let delta = g.signum();
            push_free_reduced(&mut out, -eps * (i + 1));
            push_free_reduced(&mut out, delta * i);
            push_free_reduced(&mut out, eps * (i + 1));
        } else {
            push_free_reduced(&mut out, g);
        }
    }
    for &g in &word[(q + 1)..] {
        push_free_reduced(&mut out, g);
    }
    out
}
