//! Thin `web-sys` / `wasm-bindgen` WebGPU `@compute` dispatcher (`GpuAdapter`, `GpuDevice`,
//! `GpuComputePipeline`, and double-buffered `MAP_READ` staging `GpuBuffer`s) alongside
//! `async_sim::wasm` while leaving `eframe::WebRunner` on WebGL2.

/// WebGPU `GPUBufferUsage.MAP_READ` bitflag (`0x0001`).
pub const GPU_BUFFER_USAGE_MAP_READ: u32 = 0x0001;
/// WebGPU `GPUBufferUsage.COPY_SRC` bitflag (`0x0004`).
pub const GPU_BUFFER_USAGE_COPY_SRC: u32 = 0x0004;
/// WebGPU `GPUBufferUsage.COPY_DST` bitflag (`0x0008`).
pub const GPU_BUFFER_USAGE_COPY_DST: u32 = 0x0008;
/// WebGPU `GPUBufferUsage.UNIFORM` bitflag (`0x0040`).
pub const GPU_BUFFER_USAGE_UNIFORM: u32 = 0x0040;
/// WebGPU `GPUBufferUsage.STORAGE` bitflag (`0x0080`).
pub const GPU_BUFFER_USAGE_STORAGE: u32 = 0x0080;
/// WebGPU `GPUMapMode.READ` bitflag (`0x0001`).
pub const GPU_MAP_MODE_READ: u32 = 0x0001;

/// Lifecycle state of the thin browser [`WebGpuComputeDispatcher`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WebGpuDispatcherState {
    /// Adapter/device handshake has not been requested yet.
    Uninitialized,
    /// Asynchronous `navigator.gpu.requestAdapter()` / `requestDevice()` is in flight.
    Initializing,
    /// `GpuAdapter` and `GpuDevice` are active with double-buffered `MAP_READ` staging buffers.
    Ready,
    /// `navigator.gpu` is absent or adapter acquisition failed; callers fall back to CPU/Worker.
    Unavailable,
}

/// Parameters for a single `@compute` kernel dispatch and double-buffered staging readback.
#[derive(Clone, Copy, Debug)]
pub struct WebGpuDispatchSpec<'a> {
    /// Diagnostic label attached to the `GPUShaderModule` and `GPUComputePipeline`.
    pub shader_label: &'static str,
    /// Complete `@compute` WGSL module source.
    pub wgsl_source: &'a str,
    /// Compute entry-point identifier (typically `"cs_main"`).
    pub entry_point: &'static str,
    /// Workgroup counts `[wx, wy, wz]` passed to `dispatchWorkgroups`.
    pub workgroups: [u32; 3],
    /// Uniform block payload written to `@group(0) @binding(0)` (`16`-byte aligned).
    pub uniform_bytes: &'a [u8],
    /// Input `f32` storage slice uploaded to `@group(0) @binding(1)`.
    pub input_f32: &'a [f32],
    /// Number of `f32` scalars expected back from `@group(0) @binding(2)`.
    pub output_len: usize,
}

/// Computes `ceil(dim / tile)` workgroup counts along each 3D axis (`>= 1`).
#[must_use]
pub const fn compute_workgroup_counts(dims: [u32; 3], workgroup_size: [u32; 3]) -> [u32; 3] {
    let wx = workgroup_size[0];
    let wy = workgroup_size[1];
    let wz = workgroup_size[2];
    if wx == 0 || wy == 0 || wz == 0 {
        return [1, 1, 1];
    }
    let dx = if dims[0] == 0 { 1 } else { dims[0] };
    let dy = if dims[1] == 0 { 1 } else { dims[1] };
    let dz = if dims[2] == 0 { 1 } else { dims[2] };
    [dx.div_ceil(wx), dy.div_ceil(wy), dz.div_ceil(wz)]
}

/// Thin `web-sys` / `wasm-bindgen` WebGPU `@compute` pipeline & double-buffered staging dispatcher.
pub struct WebGpuComputeDispatcher {
    #[cfg(target_arch = "wasm32")]
    inner: wasm_impl::WasmWebGpuDispatcher,
    #[cfg(not(target_arch = "wasm32"))]
    host: HostWebGpuStub,
}

impl Default for WebGpuComputeDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl WebGpuComputeDispatcher {
    /// Creates a new lazily initialized WebGPU compute dispatcher.
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(target_arch = "wasm32")]
            inner: wasm_impl::WasmWebGpuDispatcher::new(),
            #[cfg(not(target_arch = "wasm32"))]
            host: HostWebGpuStub::new(),
        }
    }

    /// Returns `true` when `window.navigator.gpu` is present in the active runtime.
    #[must_use]
    pub fn has_navigator_gpu() -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            wasm_impl::has_navigator_gpu()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::env::var("OXIDIZE_FORCE_WEBGPU_PROBE").ok().as_deref() == Some("gpu")
        }
    }

    /// Current lifecycle state (`Uninitialized`, `Initializing`, `Ready`, or `Unavailable`).
    #[must_use]
    pub fn state(&self) -> WebGpuDispatcherState {
        #[cfg(target_arch = "wasm32")]
        {
            self.inner.state()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.host.state
        }
    }

    /// Active staging ring index (`0` or `1`) for double-buffered `MAP_READ` readback.
    #[must_use]
    pub fn active_staging_slot(&self) -> usize {
        #[cfg(target_arch = "wasm32")]
        {
            self.inner.active_staging_slot()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.host.staging_slot
        }
    }

    /// Submits a non-blocking WebGPU `@compute` pass and schedules `mapAsync` staging readback.
    /// Returns `true` if a dispatch was enqueued on an active `GpuDevice`.
    pub fn submit_compute(&mut self, spec: WebGpuDispatchSpec<'_>) -> bool {
        if spec.output_len == 0 || spec.wgsl_source.is_empty() {
            return false;
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.inner.submit_compute(spec)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.host.submit_stub()
        }
    }

    /// Polls any completed `MAP_READ` staging buffer into `dst`, returning elapsed milliseconds.
    pub fn poll_readback(&mut self, dst: &mut [f32]) -> Option<f64> {
        #[cfg(target_arch = "wasm32")]
        {
            self.inner.poll_readback(dst)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = dst;
            None
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct HostWebGpuStub {
    state: WebGpuDispatcherState,
    staging_slot: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl HostWebGpuStub {
    fn new() -> Self {
        let state = if WebGpuComputeDispatcher::has_navigator_gpu() {
            WebGpuDispatcherState::Uninitialized
        } else {
            WebGpuDispatcherState::Unavailable
        };
        Self {
            state,
            staging_slot: 0,
        }
    }

    fn submit_stub(&mut self) -> bool {
        self.staging_slot ^= 1;
        if self.state == WebGpuDispatcherState::Uninitialized {
            self.state = WebGpuDispatcherState::Unavailable;
        }
        false
    }
}

#[cfg(target_arch = "wasm32")]
#[path = "webgpu_wasm.rs"]
mod wasm_impl;

#[cfg(test)]
#[path = "webgpu_tests.rs"]
mod tests;
