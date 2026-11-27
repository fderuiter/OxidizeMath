//! Deterministic `ChaCha8Rng` provider with stream splitting and word-position seeking.
// [cite:clinical_trials_statistics]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]

use crate::scalar::{
    RngScalar, sample_bounded_u64, sample_bounded_usize, sample_closed_open, sample_open,
    sample_symmetric_open,
};
use rand_chacha::ChaCha8Rng;
use rand_core::{CryptoRng, Error as RandError, RngCore, SeedableRng};

/// Standard canonical base seed for reproducible simulations across native and WASM targets.
pub const BASE_SEED: u64 = 0x0001_D1CE_CAFE_BABE;

/// Deterministic pseudo-random number generator backed by `ChaCha8Rng`.
///
/// Forbids ambient OS entropy calls while exposing ChaCha8's native 64-bit stream selector
/// ($2^{64}$ orthogonal streams of period $2^{68}$ bytes each) and 128-bit word-offset seeking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OxidizeRng {
    inner: ChaCha8Rng,
}

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename = "ChaCha8Rng")]
struct ChaCha8RngWire {
    seed: [u8; 32],
    stream: u64,
    word_pos: u128,
}

#[cfg(feature = "serde")]
impl serde::Serialize for OxidizeRng {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        ChaCha8RngWire {
            seed: self.inner.get_seed(),
            stream: self.inner.get_stream(),
            word_pos: self.inner.get_word_pos(),
        }
        .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for OxidizeRng {
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ChaCha8RngWire::deserialize(deserializer)?;
        let mut inner = <ChaCha8Rng as SeedableRng>::from_seed(wire.seed);
        inner.set_stream(wire.stream);
        inner.set_word_pos(wire.word_pos);
        Ok(Self { inner })
    }
}

impl OxidizeRng {
    /// Initializes a deterministic `OxidizeRng` from a 64-bit integer seed on stream `0`.
    #[inline]
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let rng = ChaCha8Rng::seed_from_u64(seed);
        debug_assert_eq!(rng.get_stream(), 0);
        Self { inner: rng }
    }

    /// Initializes a deterministic `OxidizeRng` from a 64-bit integer seed.
    ///
    /// Preserves inherent `OxidizeRng::from_seed(u64)` source compatibility across workspace callers;
    /// full 256-bit seed arrays are supported via [`OxidizeRng::from_seed_bytes`] and [`SeedableRng`].
    #[inline]
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        Self::new(seed)
    }

    /// Initializes a deterministic `OxidizeRng` from a full 256-bit (`[u8; 32]`) ChaCha8 key.
    #[inline]
    #[must_use]
    pub fn from_seed_bytes(seed: [u8; 32]) -> Self {
        let rng = <ChaCha8Rng as SeedableRng>::from_seed(seed);
        debug_assert_eq!(rng.get_word_pos(), 0);
        Self { inner: rng }
    }

    /// Consumes `self`, selects the given 64-bit ChaCha8 `stream`, and resets the word position to `0`.
    #[inline]
    #[must_use]
    pub fn with_stream(mut self, stream: u64) -> Self {
        self.set_stream(stream);
        self.set_word_pos(0);
        debug_assert_eq!(self.stream(), stream);
        self
    }

    /// Forks an independent child `OxidizeRng` sharing the same 256-bit key on `stream` at word `0`.
    #[inline]
    #[must_use]
    pub fn split_stream(&self, stream: u64) -> Self {
        let mut child = self.clone();
        child.set_stream(stream);
        child.set_word_pos(0);
        debug_assert_eq!(child.stream(), stream);
        child
    }

    /// Sets the active 64-bit ChaCha8 stream identifier in place.
    #[inline]
    pub fn set_stream(&mut self, stream: u64) {
        self.inner.set_stream(stream);
        debug_assert_eq!(self.inner.get_stream(), stream);
    }

    /// Returns the active 64-bit ChaCha8 stream identifier.
    #[inline]
    #[must_use]
    pub fn stream(&self) -> u64 {
        self.inner.get_stream()
    }

    /// Sets the 128-bit 32-bit-word offset within the current ChaCha8 stream.
    #[inline]
    pub fn set_word_pos(&mut self, word_offset: u128) {
        self.inner.set_word_pos(word_offset);
        debug_assert_eq!(self.inner.get_word_pos(), word_offset);
    }

    /// Returns the current 128-bit 32-bit-word offset within the active ChaCha8 stream.
    #[inline]
    #[must_use]
    pub fn word_pos(&self) -> u128 {
        self.inner.get_word_pos()
    }

    /// Advances the current ChaCha8 stream position forward by `delta_words` 32-bit words.
    #[inline]
    pub fn jump_words(&mut self, delta_words: u128) {
        let next_pos = self.inner.get_word_pos().wrapping_add(delta_words);
        self.inner.set_word_pos(next_pos);
        debug_assert_eq!(self.inner.get_word_pos(), next_pos);
    }

    /// Samples a half-open uniform scalar in `[0, 1)`.
    #[inline]
    pub fn uniform_closed_open<T: RngScalar>(&mut self) -> T {
        sample_closed_open(self)
    }

    /// Samples a strictly open uniform scalar in `(0, 1)`.
    #[inline]
    pub fn uniform_open<T: RngScalar>(&mut self) -> T {
        sample_open(self)
    }

    /// Samples a symmetric open uniform scalar in `(-1, 1)`.
    #[inline]
    pub fn uniform_symmetric<T: RngScalar>(&mut self) -> T {
        sample_symmetric_open(self)
    }

    /// Samples an unbiased integer index in `0..upper` identically on 32-bit and 64-bit targets.
    #[inline]
    pub fn bounded_usize(&mut self, upper: usize) -> usize {
        sample_bounded_usize(self, upper)
    }

    /// Samples an unbiased `u64` integer in `0..upper`.
    #[inline]
    pub fn bounded_u64(&mut self, upper: u64) -> u64 {
        sample_bounded_u64(self, upper)
    }
}

impl Default for OxidizeRng {
    #[inline]
    fn default() -> Self {
        Self::new(BASE_SEED)
    }
}

impl SeedableRng for OxidizeRng {
    type Seed = [u8; 32];

    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        Self::from_seed_bytes(seed)
    }

    #[inline]
    fn seed_from_u64(state: u64) -> Self {
        Self::new(state)
    }
}

impl RngCore for OxidizeRng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.inner.fill_bytes(dest);
    }

    #[inline]
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RandError> {
        self.inner.try_fill_bytes(dest)
    }
}

impl CryptoRng for OxidizeRng {}
