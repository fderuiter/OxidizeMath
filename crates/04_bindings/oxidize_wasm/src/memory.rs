//! Pre-allocated contiguous `f32` and `f64` linear-memory buffer handles for zero-copy JS `Float32Array` and `Float64Array` views.
// [cite:math_commons]

use crate::error::WasmSdkError;
use wasm_bindgen::prelude::*;

/// Maximum single `WasmF64Buffer` scalar capacity (`16,777,216` `f64` elements = `128 MiB`).
pub const MAX_WASM_F64_BUFFER_LEN: usize = 16_777_216;

/// Maximum single `WasmF32Buffer` scalar capacity (`33,554,432` `f32` elements = `128 MiB`).
pub const MAX_WASM_F32_BUFFER_LEN: usize = 33_554_432;

/// Pre-allocated contiguous `f32` buffer residing in WebAssembly linear memory.
///
/// JavaScript callers can construct a zero-copy typed array view over this buffer via
/// `new Float32Array(wasmLinearMemory().buffer, buf.byteOffset(), buf.len())` for direct
/// WebGPU `queue.writeBuffer` and `MAP_READ` staging transfers.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmF32Buffer {
    data: Vec<f32>,
}

#[wasm_bindgen]
impl WasmF32Buffer {
    /// Fallibly allocates a zero-initialized `f32` buffer of length `len` (`1..=MAX_WASM_F32_BUFFER_LEN`).
    #[wasm_bindgen(js_name = tryNew)]
    pub fn try_new(len: usize) -> Result<Self, WasmSdkError> {
        if len == 0 || len > MAX_WASM_F32_BUFFER_LEN {
            return Err(WasmSdkError::invalid_argument(
                "E_BUFFER_CAPACITY",
                format!("WasmF32Buffer length {len} must be within 1..={MAX_WASM_F32_BUFFER_LEN}"),
                MAX_WASM_F32_BUFFER_LEN,
                len,
            ));
        }
        let mut data = Vec::new();
        data.try_reserve_exact(len).map_err(|_| {
            WasmSdkError::invalid_argument(
                "E_BUFFER_ALLOC",
                "Failed to reserve contiguous f32 buffer in WASM linear memory",
                len,
                0,
            )
        })?;
        data.resize(len, 0.0);
        Ok(Self { data })
    }

    /// Byte offset of element `0` inside WebAssembly linear memory (`WebAssembly.Memory.buffer`).
    #[wasm_bindgen(js_name = byteOffset)]
    #[must_use]
    pub fn byte_offset(&self) -> usize {
        self.data.as_ptr() as usize
    }

    /// Number of `f32` elements in this buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when this buffer has zero elements.
    #[wasm_bindgen(js_name = isEmpty)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Fills every element in the pre-allocated buffer with `value`.
    pub fn fill(&mut self, value: f32) {
        self.data.fill(value);
    }

    /// Copies `src` into the leading `src.len()` elements without reallocating.
    #[wasm_bindgen(js_name = copyFromSlice)]
    pub fn copy_from_slice(&mut self, src: &[f32]) -> Result<(), WasmSdkError> {
        if src.len() > self.data.len() {
            return Err(WasmSdkError::invalid_argument(
                "E_BUFFER_OVERFLOW",
                "Source slice exceeds WasmF32Buffer capacity",
                self.data.len(),
                src.len(),
            ));
        }
        self.data[..src.len()].copy_from_slice(src);
        Ok(())
    }

    /// Copies `dst.len()` elements from this buffer into caller-provided `dst` (`Float32Array`).
    #[wasm_bindgen(js_name = copyIntoSlice)]
    pub fn copy_into_slice(&self, dst: &mut [f32]) -> Result<(), WasmSdkError> {
        if dst.len() > self.data.len() {
            return Err(WasmSdkError::invalid_argument(
                "E_BUFFER_UNDERFLOW",
                "Destination slice exceeds WasmF32Buffer length",
                self.data.len(),
                dst.len(),
            ));
        }
        dst.copy_from_slice(&self.data[..dst.len()]);
        Ok(())
    }
}

impl WasmF32Buffer {
    /// Immutable Rust slice view over the linear-memory `f32` buffer.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Mutable Rust slice view over the linear-memory `f32` buffer.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

/// Pre-allocated contiguous `f64` buffer residing in WebAssembly linear memory.
///
/// JavaScript callers can construct a zero-copy typed array view over this buffer via
/// `new Float64Array(wasmLinearMemory().buffer, buf.byteOffset(), buf.len())`.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmF64Buffer {
    data: Vec<f64>,
}

#[wasm_bindgen]
impl WasmF64Buffer {
    /// Fallibly allocates a zero-initialized `f64` buffer of length `len` (`1..=MAX_WASM_F64_BUFFER_LEN`).
    #[wasm_bindgen(js_name = tryNew)]
    pub fn try_new(len: usize) -> Result<Self, WasmSdkError> {
        if len == 0 || len > MAX_WASM_F64_BUFFER_LEN {
            return Err(WasmSdkError::invalid_argument(
                "E_BUFFER_CAPACITY",
                format!("WasmF64Buffer length {len} must be within 1..={MAX_WASM_F64_BUFFER_LEN}"),
                MAX_WASM_F64_BUFFER_LEN,
                len,
            ));
        }
        let mut data = Vec::new();
        data.try_reserve_exact(len).map_err(|_| {
            WasmSdkError::invalid_argument(
                "E_BUFFER_ALLOC",
                "Failed to reserve contiguous f64 buffer in WASM linear memory",
                len,
                0,
            )
        })?;
        data.resize(len, 0.0);
        Ok(Self { data })
    }

    /// Byte offset of element `0` inside WebAssembly linear memory (`WebAssembly.Memory.buffer`).
    #[wasm_bindgen(js_name = byteOffset)]
    #[must_use]
    pub fn byte_offset(&self) -> usize {
        self.data.as_ptr() as usize
    }

    /// Number of `f64` elements in this buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when this buffer has zero elements.
    #[wasm_bindgen(js_name = isEmpty)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Fills every element in the pre-allocated buffer with `value`.
    pub fn fill(&mut self, value: f64) {
        self.data.fill(value);
    }

    /// Copies `src` into the leading `src.len()` elements without reallocating.
    #[wasm_bindgen(js_name = copyFromSlice)]
    pub fn copy_from_slice(&mut self, src: &[f64]) -> Result<(), WasmSdkError> {
        if src.len() > self.data.len() {
            return Err(WasmSdkError::invalid_argument(
                "E_BUFFER_OVERFLOW",
                "Source slice exceeds WasmF64Buffer capacity",
                self.data.len(),
                src.len(),
            ));
        }
        self.data[..src.len()].copy_from_slice(src);
        Ok(())
    }

    /// Copies `dst.len()` elements from this buffer into caller-provided `dst` (`Float64Array`).
    #[wasm_bindgen(js_name = copyIntoSlice)]
    pub fn copy_into_slice(&self, dst: &mut [f64]) -> Result<(), WasmSdkError> {
        if dst.len() > self.data.len() {
            return Err(WasmSdkError::invalid_argument(
                "E_BUFFER_UNDERFLOW",
                "Destination slice exceeds WasmF64Buffer length",
                self.data.len(),
                dst.len(),
            ));
        }
        dst.copy_from_slice(&self.data[..dst.len()]);
        Ok(())
    }
}

impl WasmF64Buffer {
    /// Immutable Rust slice view over the linear-memory buffer.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Mutable Rust slice view over the linear-memory buffer.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }
}
