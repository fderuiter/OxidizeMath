//! Canonical IEEE-754 trajectory digest primitives for cross-platform determinism gates.
//!
//! Normalizes signed zeros (`-0.0` and `+0.0`) to canonical positive zero bits and collapses all
//! quiet/signaling `NaN` bit patterns to a single canonical quiet `NaN` representation before
//! hashing little-endian byte representations with pure-Rust SHA-256.

/// Canonical IEEE-754 binary64 quiet `NaN` bit pattern (`0x7ff8_0000_0000_0000`).
pub const CANONICAL_F64_NAN_BITS: u64 = 0x7ff8_0000_0000_0000;

/// Canonical IEEE-754 binary32 quiet `NaN` bit pattern (`0x7fc0_0000`).
pub const CANONICAL_F32_NAN_BITS: u32 = 0x7fc0_0000;

const SHA256_INIT: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

const SHA256_K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// Returns canonical IEEE-754 `f64` bits with `-0.0` mapped to `+0.0` and all `NaN` payloads
/// collapsed to [`CANONICAL_F64_NAN_BITS`].
#[inline]
#[must_use]
pub fn canonical_f64_bits(value: f64) -> u64 {
    if value.is_nan() {
        CANONICAL_F64_NAN_BITS
    } else if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

/// Returns canonical IEEE-754 `f32` bits with `-0.0` mapped to `+0.0` and all `NaN` payloads
/// collapsed to [`CANONICAL_F32_NAN_BITS`].
#[inline]
#[must_use]
pub fn canonical_f32_bits(value: f32) -> u32 {
    if value.is_nan() {
        CANONICAL_F32_NAN_BITS
    } else if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

fn compress_block(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0_u32; 64];
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

    for (slot, delta) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *slot = slot.wrapping_add(delta);
    }
}

/// Streaming, allocation-free SHA-256 accumulator for canonical stochastic trajectory digests.
#[derive(Clone, Debug)]
pub struct TrajectoryDigest {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_bytes: u64,
}

impl Default for TrajectoryDigest {
    fn default() -> Self {
        Self::new()
    }
}

impl TrajectoryDigest {
    /// Creates an empty canonical trajectory digest accumulator.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: SHA256_INIT,
            buf: [0_u8; 64],
            buf_len: 0,
            total_bytes: 0,
        }
    }

    /// Feeds raw bytes into the digest without heap allocation.
    pub fn update_bytes(&mut self, mut bytes: &[u8]) {
        self.total_bytes = self.total_bytes.wrapping_add(bytes.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(bytes.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&bytes[..take]);
            self.buf_len += take;
            bytes = &bytes[take..];
            if self.buf_len == 64 {
                compress_block(&mut self.state, &self.buf);
                self.buf_len = 0;
            }
        }
        while bytes.len() >= 64 {
            let block: &[u8; 64] = bytes[..64].try_into().expect("64-byte slice");
            compress_block(&mut self.state, block);
            bytes = &bytes[64..];
        }
        if !bytes.is_empty() {
            self.buf[..bytes.len()].copy_from_slice(bytes);
            self.buf_len = bytes.len();
        }
    }

    /// Feeds a canonicalized IEEE-754 `f64` value in little-endian byte order.
    #[inline]
    pub fn update_f64(&mut self, value: f64) {
        self.update_bytes(&canonical_f64_bits(value).to_le_bytes());
    }

    /// Feeds a slice of `f64` values using [`canonical_f64_bits`].
    pub fn update_f64_slice(&mut self, values: &[f64]) {
        for &v in values {
            self.update_f64(v);
        }
    }

    /// Feeds a canonicalized IEEE-754 `f32` value in little-endian byte order.
    #[inline]
    pub fn update_f32(&mut self, value: f32) {
        self.update_bytes(&canonical_f32_bits(value).to_le_bytes());
    }

    /// Feeds a slice of `f32` values using [`canonical_f32_bits`].
    pub fn update_f32_slice(&mut self, values: &[f32]) {
        for &v in values {
            self.update_f32(v);
        }
    }

    /// Feeds an unsigned 64-bit integer in little-endian byte order.
    #[inline]
    pub fn update_u64(&mut self, value: u64) {
        self.update_bytes(&value.to_le_bytes());
    }

    /// Feeds a signed 64-bit integer in little-endian byte order.
    #[inline]
    pub fn update_i64(&mut self, value: i64) {
        self.update_bytes(&value.to_le_bytes());
    }

    /// Feeds an unsigned 32-bit integer in little-endian byte order.
    #[inline]
    pub fn update_u32(&mut self, value: u32) {
        self.update_bytes(&value.to_le_bytes());
    }

    /// Feeds a signed 32-bit integer in little-endian byte order.
    #[inline]
    pub fn update_i32(&mut self, value: i32) {
        self.update_bytes(&value.to_le_bytes());
    }

    /// Feeds a `usize` widened to 64-bit little-endian so `wasm32` and 64-bit hosts match.
    #[inline]
    pub fn update_usize(&mut self, value: usize) {
        self.update_u64(value as u64);
    }

    /// Feeds a boolean flag as a single canonical byte (`0` or `1`).
    #[inline]
    pub fn update_bool(&mut self, value: bool) {
        self.update_bytes(&[u8::from(value)]);
    }

    /// Finalizes a copy of the current accumulator into a 32-byte SHA-256 digest.
    #[must_use]
    pub fn finish_bytes(&self) -> [u8; 32] {
        let mut clone = self.clone();
        let bit_len = clone.total_bytes.wrapping_mul(8);
        clone.update_bytes(&[0x80]);
        while clone.buf_len != 56 {
            let pad_zeros = if clone.buf_len < 56 {
                56 - clone.buf_len
            } else {
                64 - clone.buf_len + 56
            };
            let zeros = [0_u8; 64];
            clone.update_bytes(&zeros[..pad_zeros]);
        }
        clone.update_bytes(&bit_len.to_be_bytes());
        debug_assert_eq!(clone.buf_len, 0);

        let mut out = [0_u8; 32];
        for (i, word) in clone.state.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    /// Finalizes the accumulator and formats a lowercase 64-character hexadecimal digest.
    #[must_use]
    pub fn finish_hex(&self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let raw = self.finish_bytes();
        let mut s = String::with_capacity(64);
        for byte in raw {
            s.push(HEX[(byte >> 4) as usize] as char);
            s.push(HEX[(byte & 0x0f) as usize] as char);
        }
        s
    }
}

/// Computes the canonical 64-character lowercase SHA-256 hex digest of an `f64` trajectory slice.
#[must_use]
pub fn digest_f64_slice(values: &[f64]) -> String {
    let mut d = TrajectoryDigest::new();
    d.update_f64_slice(values);
    d.finish_hex()
}

/// Computes the canonical 64-character lowercase SHA-256 hex digest of an `f32` trajectory slice.
#[must_use]
pub fn digest_f32_slice(values: &[f32]) -> String {
    let mut d = TrajectoryDigest::new();
    d.update_f32_slice(values);
    d.finish_hex()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_nist_vectors() {
        let empty = TrajectoryDigest::new().finish_hex();
        assert_eq!(
            empty,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let mut abc = TrajectoryDigest::new();
        abc.update_bytes(b"abc");
        assert_eq!(
            abc.finish_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn canonicalizes_signed_zero_and_nan_payloads() {
        let neg_nan_64 = f64::from_bits(0xfff8_0123_4567_89ab);
        let sig_nan_64 = f64::from_bits(0x7ff0_0000_0000_0001);
        assert_eq!(canonical_f64_bits(-0.0), canonical_f64_bits(0.0));
        assert_eq!(canonical_f64_bits(f64::NAN), CANONICAL_F64_NAN_BITS);
        assert_eq!(canonical_f64_bits(neg_nan_64), CANONICAL_F64_NAN_BITS);
        assert_eq!(canonical_f64_bits(sig_nan_64), CANONICAL_F64_NAN_BITS);

        let a = digest_f64_slice(&[0.0, f64::NAN, 1.5, -2.25]);
        let b = digest_f64_slice(&[-0.0, neg_nan_64, 1.5, -2.25]);
        assert_eq!(a, b);

        let neg_nan_32 = f32::from_bits(0xffc0_1234);
        assert_eq!(
            digest_f32_slice(&[0.0, f32::NAN, -3.5]),
            digest_f32_slice(&[-0.0, neg_nan_32, -3.5])
        );
    }
}
