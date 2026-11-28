//! Off-main-thread Web Worker simulation bridge supporting `SharedArrayBuffer` zero-copy frames
//! when `crossOriginIsolated == true` and transferable `ArrayBuffer` `postMessage` frames on
//! standard non-isolated static hosts (such as GitHub Pages).

use crate::web_io::PlatformInstant;
use math_commons::math_kernel::types::flatten_3d_index;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};

/// Runtime transport selected for off-main-thread volumetric simulation stepping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerTransportMode {
    /// `crossOriginIsolated == true` with `SharedArrayBuffer` + `Atomics` zero-copy double buffering.
    SharedArrayBufferRing,
    /// Dedicated `Blob`-backed `Worker` exchanging transferable `Float32Array` `ArrayBuffer`s.
    DedicatedWorkerPostMessage,
    /// Cooperative main-thread slice fallback when `Worker` construction is unavailable.
    CooperativeMainThread,
    /// Native OS background `std::thread` channel bridge on desktop targets.
    NativeBackgroundThread,
}

impl WorkerTransportMode {
    /// Concise telemetry label describing the active execution transport.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SharedArrayBufferRing => "SAB Worker",
            Self::DedicatedWorkerPostMessage => "Web Worker",
            Self::CooperativeMainThread => "Coop Slice",
            Self::NativeBackgroundThread => "Native Worker",
        }
    }
}

/// Capability snapshot probed from the active browser or desktop runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WebWorkerCapabilities {
    /// Whether `self.crossOriginIsolated === true` in the current browsing context.
    pub cross_origin_isolated: bool,
    /// Whether `typeof SharedArrayBuffer !== "undefined"` in the global scope.
    pub shared_array_buffer_available: bool,
    /// Whether `typeof Worker !== "undefined"` and `Blob` URLs are available.
    pub dedicated_worker_available: bool,
    /// Transport mode selected from the probed capabilities.
    pub selected_transport: WorkerTransportMode,
}

/// Pure selection rule mapping runtime isolation/API flags to a [`WorkerTransportMode`].
#[must_use]
pub const fn select_wasm_transport(
    cross_origin_isolated: bool,
    shared_array_buffer_available: bool,
    dedicated_worker_available: bool,
) -> WorkerTransportMode {
    if cross_origin_isolated && shared_array_buffer_available && dedicated_worker_available {
        WorkerTransportMode::SharedArrayBufferRing
    } else if dedicated_worker_available {
        WorkerTransportMode::DedicatedWorkerPostMessage
    } else {
        WorkerTransportMode::CooperativeMainThread
    }
}

/// Probes `crossOriginIsolated`, `SharedArrayBuffer`, and `Worker` support in the current runtime.
#[must_use]
pub fn probe_web_worker_capabilities() -> WebWorkerCapabilities {
    #[cfg(target_arch = "wasm32")]
    {
        wasm_impl::probe_capabilities_wasm()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        WebWorkerCapabilities {
            cross_origin_isolated: false,
            shared_array_buffer_available: false,
            dedicated_worker_available: true,
            selected_transport: WorkerTransportMode::NativeBackgroundThread,
        }
    }
}

/// Parameters for a 3D Schnakenberg reaction-diffusion batch step on a cubic `dim^3` grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumetricStepJob {
    /// Cubic grid side length (`16..=128`).
    pub dim: usize,
    /// Explicit integration step `dt`.
    pub dt: f32,
    /// Activator diffusion coefficient `D_u`.
    pub du: f32,
    /// Inhibitor diffusion coefficient `D_v`.
    pub dv: f32,
    /// Schnakenberg feed rate `a`.
    pub a: f32,
    /// Schnakenberg feed rate `b`.
    pub b: f32,
    /// Number of explicit substeps executed per worker dispatch (`1..=8`).
    pub substeps: u32,
}

/// Executes `job.substeps` 7-point periodic 3D Schnakenberg sweeps in `f32` using `flatten_3d_index`.
pub fn step_schnakenberg_3d_f32(
    job: VolumetricStepJob,
    u: &mut [f32],
    v: &mut [f32],
    u_next: &mut [f32],
    v_next: &mut [f32],
) {
    let d = job.dim;
    let total = d.saturating_mul(d).saturating_mul(d);
    if d < 3 || u.len() < total || v.len() < total || u_next.len() < total || v_next.len() < total {
        return;
    }
    let steps = job.substeps.clamp(1, 8);
    for _ in 0..steps {
        sweep_single_schnakenberg_substep(job, &u[..total], &v[..total], u_next, v_next);
        u[..total].copy_from_slice(&u_next[..total]);
        v[..total].copy_from_slice(&v_next[..total]);
    }
}

fn sweep_single_schnakenberg_substep(
    job: VolumetricStepJob,
    u: &[f32],
    v: &[f32],
    u_out: &mut [f32],
    v_out: &mut [f32],
) {
    let d = job.dim;
    for z in 0..d {
        let zm = if z == 0 { d - 1 } else { z - 1 };
        let zp = if z + 1 == d { 0 } else { z + 1 };
        for y in 0..d {
            let ym = if y == 0 { d - 1 } else { y - 1 };
            let yp = if y + 1 == d { 0 } else { y + 1 };
            for x in 0..d {
                let xm = if x == 0 { d - 1 } else { x - 1 };
                let xp = if x + 1 == d { 0 } else { x + 1 };
                let idx = flatten_3d_index(x, y, z, d, d);
                let uc = u[idx];
                let vc = v[idx];
                let lap_u = u[flatten_3d_index(xm, y, z, d, d)]
                    + u[flatten_3d_index(xp, y, z, d, d)]
                    + u[flatten_3d_index(x, ym, z, d, d)]
                    + u[flatten_3d_index(x, yp, z, d, d)]
                    + u[flatten_3d_index(x, y, zm, d, d)]
                    + u[flatten_3d_index(x, y, zp, d, d)]
                    - 6.0 * uc;
                let lap_v = v[flatten_3d_index(xm, y, z, d, d)]
                    + v[flatten_3d_index(xp, y, z, d, d)]
                    + v[flatten_3d_index(x, ym, z, d, d)]
                    + v[flatten_3d_index(x, yp, z, d, d)]
                    + v[flatten_3d_index(x, y, zm, d, d)]
                    + v[flatten_3d_index(x, y, zp, d, d)]
                    - 6.0 * vc;
                let u2v = uc * uc * vc;
                let du_dt = job.du * lap_u + (job.a - uc + u2v);
                let dv_dt = job.dv * lap_v + (job.b - u2v);
                u_out[idx] = (uc + job.dt * du_dt).clamp(0.0, 12.0);
                v_out[idx] = (vc + job.dt * dv_dt).clamp(0.0, 12.0);
            }
        }
    }
}

/// Non-blocking off-main-thread bridge for high-resolution 3D volumetric stepping.
pub struct VolumetricWorkerBridge {
    caps: WebWorkerCapabilities,
    #[cfg(target_arch = "wasm32")]
    inner: wasm_impl::WasmVolumetricWorker,
    #[cfg(not(target_arch = "wasm32"))]
    inner: NativeVolumetricWorker,
}

impl Default for VolumetricWorkerBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl VolumetricWorkerBridge {
    /// Creates a new off-main-thread worker bridge using probed runtime capabilities.
    #[must_use]
    pub fn new() -> Self {
        let caps = probe_web_worker_capabilities();
        #[cfg(target_arch = "wasm32")]
        let (caps, inner) = wasm_impl::WasmVolumetricWorker::spawn(caps);
        #[cfg(not(target_arch = "wasm32"))]
        let inner = NativeVolumetricWorker::spawn();
        Self { caps, inner }
    }

    /// Probed browser/host capabilities and active transport mode.
    #[must_use]
    pub const fn capabilities(&self) -> WebWorkerCapabilities {
        self.caps
    }

    /// Active worker transport mode.
    #[must_use]
    pub const fn transport_mode(&self) -> WorkerTransportMode {
        self.caps.selected_transport
    }

    /// Returns `true` while an asynchronous worker step is currently in flight.
    #[must_use]
    pub fn is_step_in_flight(&self) -> bool {
        self.inner.is_in_flight()
    }

    /// Submits a non-blocking 3D Turing step to the background worker.
    /// Returns `false` without blocking if a previous step is still in flight.
    pub fn submit_turing_step(&mut self, job: VolumetricStepJob, u: &[f32], v: &[f32]) -> bool {
        self.inner.submit(job, u, v)
    }

    /// Polls for a completed step without blocking, copying `u` and `v` into the destination slices
    /// and returning the elapsed step duration in milliseconds when ready.
    pub fn poll_completed_step(&mut self, u_out: &mut [f32], v_out: &mut [f32]) -> Option<f64> {
        self.inner.poll(u_out, v_out)
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeJobPayload {
    job: VolumetricStepJob,
    u: Vec<f32>,
    v: Vec<f32>,
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeJobResult {
    u: Vec<f32>,
    v: Vec<f32>,
    elapsed_ms: f64,
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeVolumetricWorker {
    tx: SyncSender<NativeJobPayload>,
    rx: Receiver<NativeJobResult>,
    recycled_u: Vec<f32>,
    recycled_v: Vec<f32>,
    in_flight: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl NativeVolumetricWorker {
    fn spawn() -> Self {
        let (tx, job_rx) = mpsc::sync_channel::<NativeJobPayload>(1);
        let (res_tx, rx) = mpsc::sync_channel::<NativeJobResult>(1);
        let _ = std::thread::Builder::new()
            .name("volumetric-3d-worker".into())
            .spawn(move || {
                let mut u_scratch = Vec::new();
                let mut v_scratch = Vec::new();
                while let Ok(mut payload) = job_rx.recv() {
                    let t0 = PlatformInstant::now();
                    let n = payload.u.len();
                    if u_scratch.len() < n {
                        u_scratch.resize(n, 0.0);
                        v_scratch.resize(n, 0.0);
                    }
                    step_schnakenberg_3d_f32(
                        payload.job,
                        &mut payload.u,
                        &mut payload.v,
                        &mut u_scratch[..n],
                        &mut v_scratch[..n],
                    );
                    let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;
                    if res_tx
                        .send(NativeJobResult {
                            u: payload.u,
                            v: payload.v,
                            elapsed_ms,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            });
        Self {
            tx,
            rx,
            recycled_u: Vec::new(),
            recycled_v: Vec::new(),
            in_flight: false,
        }
    }

    const fn is_in_flight(&self) -> bool {
        self.in_flight
    }

    fn submit(&mut self, job: VolumetricStepJob, u: &[f32], v: &[f32]) -> bool {
        if self.in_flight || u.len() != v.len() {
            return false;
        }
        let mut u_buf = std::mem::take(&mut self.recycled_u);
        let mut v_buf = std::mem::take(&mut self.recycled_v);
        u_buf.resize(u.len(), 0.0);
        v_buf.resize(v.len(), 0.0);
        u_buf.copy_from_slice(u);
        v_buf.copy_from_slice(v);
        match self.tx.try_send(NativeJobPayload {
            job,
            u: u_buf,
            v: v_buf,
        }) {
            Ok(()) => {
                self.in_flight = true;
                true
            }
            Err(TrySendError::Full(p) | TrySendError::Disconnected(p)) => {
                self.recycled_u = p.u;
                self.recycled_v = p.v;
                false
            }
        }
    }

    fn poll(&mut self, u_out: &mut [f32], v_out: &mut [f32]) -> Option<f64> {
        let Ok(done) = self.rx.try_recv() else {
            return None;
        };
        self.in_flight = false;
        let len = done.u.len().min(u_out.len()).min(v_out.len());
        u_out[..len].copy_from_slice(&done.u[..len]);
        v_out[..len].copy_from_slice(&done.v[..len]);
        self.recycled_u = done.u;
        self.recycled_v = done.v;
        Some(done.elapsed_ms)
    }
}

#[cfg(target_arch = "wasm32")]
#[path = "web_worker_wasm.rs"]
mod wasm_impl;

#[cfg(test)]
#[path = "web_worker_tests.rs"]
mod tests;
