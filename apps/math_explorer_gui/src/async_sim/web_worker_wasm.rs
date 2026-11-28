//! Browser `wasm32` `Blob` Web Worker + `SharedArrayBuffer` / transferable `ArrayBuffer` backend
//! for [`super::VolumetricWorkerBridge`].

use super::{
    select_wasm_transport, step_schnakenberg_3d_f32, PlatformInstant, VolumetricStepJob,
    WebWorkerCapabilities, WorkerTransportMode,
};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};

const WORKER_JS_SOURCE: &str = r#"
let sabFloats = null;
function stepSchnakenberg(d, dt, du, dv, a, b, substeps, u, v, uOut, vOut) {
  const sMax = Math.max(1, Math.min(8, substeps | 0));
  const slice = d * d;
  for (let s = 0; s < sMax; s++) {
    for (let z = 0; z < d; z++) {
      const zm = (z === 0 ? d - 1 : z - 1) * slice;
      const zc = z * slice;
      const zp = (z + 1 === d ? 0 : z + 1) * slice;
      for (let y = 0; y < d; y++) {
        const ym = (y === 0 ? d - 1 : y - 1) * d;
        const yc = y * d;
        const yp = (y + 1 === d ? 0 : y + 1) * d;
        for (let x = 0; x < d; x++) {
          const xm = x === 0 ? d - 1 : x - 1;
          const xp = x + 1 === d ? 0 : x + 1;
          const idx = zc + yc + x;
          const uc = u[idx];
          const vc = v[idx];
          const lapU = u[zc + yc + xm] + u[zc + yc + xp] + u[zc + ym + x] + u[zc + yp + x] + u[zm + yc + x] + u[zp + yc + x] - 6.0 * uc;
          const lapV = v[zc + yc + xm] + v[zc + yc + xp] + v[zc + ym + x] + v[zc + yp + x] + v[zm + yc + x] + v[zp + yc + x] - 6.0 * vc;
          const u2v = uc * uc * vc;
          let un = uc + dt * (du * lapU + (a - uc + u2v));
          let vn = vc + dt * (dv * lapV + (b - u2v));
          uOut[idx] = un < 0.0 ? 0.0 : (un > 12.0 ? 12.0 : un);
          vOut[idx] = vn < 0.0 ? 0.0 : (vn > 12.0 ? 12.0 : vn);
        }
      }
    }
    u.set(uOut);
    v.set(vOut);
  }
}
self.onmessage = function(ev) {
  const m = ev.data;
  const t0 = performance.now();
  if (m.kind === "init_sab") {
    sabFloats = new Float32Array(m.sab);
    return;
  }
  if (m.kind === "step_sab" && sabFloats) {
    const n = m.len | 0;
    const u = sabFloats.subarray(0, n);
    const v = sabFloats.subarray(n, 2 * n);
    const uTmp = sabFloats.subarray(2 * n, 3 * n);
    const vTmp = sabFloats.subarray(3 * n, 4 * n);
    stepSchnakenberg(m.dim, m.dt, m.du, m.dv, m.a, m.b, m.substeps, u, v, uTmp, vTmp);
    self.postMessage({ kind: "done_sab", elapsedMs: performance.now() - t0 });
    return;
  }
  if (m.kind === "step_transfer") {
    const u = new Float32Array(m.uBuf);
    const v = new Float32Array(m.vBuf);
    const uTmp = new Float32Array(u.length);
    const vTmp = new Float32Array(v.length);
    stepSchnakenberg(m.dim, m.dt, m.du, m.dv, m.a, m.b, m.substeps, u, v, uTmp, vTmp);
    self.postMessage({ kind: "done_transfer", uBuf: u.buffer, vBuf: v.buffer, elapsedMs: performance.now() - t0 }, [u.buffer, v.buffer]);
  }
};
"#;

pub(super) fn probe_capabilities_wasm() -> WebWorkerCapabilities {
    let global = js_sys::global();
    let cross_origin_isolated =
        js_sys::Reflect::get(&global, &JsValue::from_str("crossOriginIsolated"))
            .ok()
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
    let shared_array_buffer_available =
        js_sys::Reflect::get(&global, &JsValue::from_str("SharedArrayBuffer"))
            .is_ok_and(|v| !v.is_undefined());
    let dedicated_worker_available = js_sys::Reflect::get(&global, &JsValue::from_str("Worker"))
        .is_ok_and(|v| !v.is_undefined())
        && js_sys::Reflect::get(&global, &JsValue::from_str("Blob"))
            .is_ok_and(|v| !v.is_undefined());
    let selected_transport = select_wasm_transport(
        cross_origin_isolated,
        shared_array_buffer_available,
        dedicated_worker_available,
    );
    WebWorkerCapabilities {
        cross_origin_isolated,
        shared_array_buffer_available,
        dedicated_worker_available,
        selected_transport,
    }
}

struct CompletedStep {
    u: Vec<f32>,
    v: Vec<f32>,
    sab_len: usize,
    elapsed_ms: f64,
}

pub(super) struct WasmVolumetricWorker {
    worker: Option<web_sys::Worker>,
    _onmessage: Option<Closure<dyn FnMut(web_sys::MessageEvent)>>,
    completed: Rc<RefCell<Option<CompletedStep>>>,
    sab: Option<(js_sys::SharedArrayBuffer, usize)>,
    fallback_u: Vec<f32>,
    fallback_v: Vec<f32>,
    in_flight: bool,
    worker_attempted: bool,
    transport: WorkerTransportMode,
}

impl WasmVolumetricWorker {
    pub(super) fn spawn(caps: WebWorkerCapabilities) -> (WebWorkerCapabilities, Self) {
        let worker_attempted = !caps.dedicated_worker_available;
        let transport = caps.selected_transport;
        (
            caps,
            Self {
                worker: None,
                _onmessage: None,
                completed: Rc::new(RefCell::new(None)),
                sab: None,
                fallback_u: Vec::new(),
                fallback_v: Vec::new(),
                in_flight: false,
                worker_attempted,
                transport,
            },
        )
    }

    fn ensure_worker_spawned(&mut self) {
        if self.worker_attempted {
            return;
        }
        self.worker_attempted = true;
        let Ok(worker) = build_blob_worker() else {
            self.transport = WorkerTransportMode::CooperativeMainThread;
            return;
        };
        let slot = Rc::clone(&self.completed);
        let onmsg = Closure::wrap(Box::new(move |ev: web_sys::MessageEvent| {
            handle_worker_message(&slot, ev);
        }) as Box<dyn FnMut(web_sys::MessageEvent)>);
        worker.set_onmessage(Some(onmsg.as_ref().unchecked_ref()));
        self.worker = Some(worker);
        self._onmessage = Some(onmsg);
    }

    pub(super) const fn is_in_flight(&self) -> bool {
        self.in_flight
    }

    pub(super) fn submit(&mut self, job: VolumetricStepJob, u: &[f32], v: &[f32]) -> bool {
        if self.in_flight || u.len() != v.len() || u.is_empty() {
            return false;
        }
        self.ensure_worker_spawned();
        if self.worker.is_none() {
            return self.run_cooperative_fallback(job, u, v);
        }
        if self.transport == WorkerTransportMode::SharedArrayBufferRing
            && self.submit_sab_step(job, u, v)
        {
            self.in_flight = true;
            return true;
        }
        let u_arr = js_sys::Float32Array::from(u);
        let v_arr = js_sys::Float32Array::from(v);
        let u_buf = u_arr.buffer();
        let v_buf = v_arr.buffer();
        let msg = build_job_object("step_transfer", job);
        let _ = js_sys::Reflect::set(&msg, &JsValue::from_str("uBuf"), &u_buf);
        let _ = js_sys::Reflect::set(&msg, &JsValue::from_str("vBuf"), &v_buf);
        let transfer = js_sys::Array::of2(&u_buf, &v_buf);
        let post_ok = self
            .worker
            .as_ref()
            .is_some_and(|w| w.post_message_with_transfer(&msg, &transfer).is_ok());
        if post_ok {
            self.in_flight = true;
            true
        } else {
            self.run_cooperative_fallback(job, u, v)
        }
    }

    fn submit_sab_step(&mut self, job: VolumetricStepJob, u: &[f32], v: &[f32]) -> bool {
        let Some(worker) = &self.worker else {
            return false;
        };
        let n = u.len();
        let needed_bytes = (n * 4 * std::mem::size_of::<f32>()) as u32;
        let needs_alloc = self
            .sab
            .as_ref()
            .is_none_or(|(s, cap_n)| *cap_n < n || s.byte_length() < needed_bytes);
        if needs_alloc {
            let sab = js_sys::SharedArrayBuffer::new(needed_bytes);
            let init_msg = js_sys::Object::new();
            let _ = js_sys::Reflect::set(
                &init_msg,
                &JsValue::from_str("kind"),
                &JsValue::from_str("init_sab"),
            );
            let _ = js_sys::Reflect::set(&init_msg, &JsValue::from_str("sab"), &sab);
            if worker.post_message(&init_msg).is_err() {
                return false;
            }
            self.sab = Some((sab, n));
        }
        let Some((sab, _)) = &self.sab else {
            return false;
        };
        let view = js_sys::Float32Array::new(sab);
        view.subarray(0, n as u32).copy_from(u);
        view.subarray(n as u32, (2 * n) as u32).copy_from(v);
        let msg = build_job_object("step_sab", job);
        let _ = js_sys::Reflect::set(
            &msg,
            &JsValue::from_str("len"),
            &JsValue::from_f64(n as f64),
        );
        worker.post_message(&msg).is_ok()
    }

    fn run_cooperative_fallback(&mut self, job: VolumetricStepJob, u: &[f32], v: &[f32]) -> bool {
        let t0 = PlatformInstant::now();
        let n = u.len();
        let mut u_work = u.to_vec();
        let mut v_work = v.to_vec();
        self.fallback_u.resize(n, 0.0);
        self.fallback_v.resize(n, 0.0);
        step_schnakenberg_3d_f32(
            job,
            &mut u_work,
            &mut v_work,
            &mut self.fallback_u,
            &mut self.fallback_v,
        );
        *self.completed.borrow_mut() = Some(CompletedStep {
            u: u_work,
            v: v_work,
            sab_len: 0,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1_000.0,
        });
        self.in_flight = true;
        true
    }

    pub(super) fn poll(&mut self, u_out: &mut [f32], v_out: &mut [f32]) -> Option<f64> {
        let done = self.completed.borrow_mut().take()?;
        self.in_flight = false;
        if done.sab_len > 0 {
            if let Some((sab, _)) = &self.sab {
                let n = u_out.len().min(v_out.len());
                let view = js_sys::Float32Array::new(sab);
                view.subarray(0, n as u32).copy_to(&mut u_out[..n]);
                view.subarray(n as u32, (2 * n) as u32)
                    .copy_to(&mut v_out[..n]);
            }
        } else {
            let n = done.u.len().min(u_out.len()).min(v_out.len());
            u_out[..n].copy_from_slice(&done.u[..n]);
            v_out[..n].copy_from_slice(&done.v[..n]);
        }
        Some(done.elapsed_ms)
    }
}

impl Drop for WasmVolumetricWorker {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.terminate();
        }
    }
}

fn handle_worker_message(slot: &Rc<RefCell<Option<CompletedStep>>>, ev: web_sys::MessageEvent) {
    let data = ev.data();
    let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    let elapsed_ms = js_sys::Reflect::get(&data, &JsValue::from_str("elapsedMs"))
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    if kind == "done_transfer" {
        let Ok(u_ab) = js_sys::Reflect::get(&data, &JsValue::from_str("uBuf")) else {
            return;
        };
        let Ok(v_ab) = js_sys::Reflect::get(&data, &JsValue::from_str("vBuf")) else {
            return;
        };
        let u_arr = js_sys::Float32Array::new(&u_ab);
        let v_arr = js_sys::Float32Array::new(&v_ab);
        *slot.borrow_mut() = Some(CompletedStep {
            u: u_arr.to_vec(),
            v: v_arr.to_vec(),
            sab_len: 0,
            elapsed_ms,
        });
    } else if kind == "done_sab" {
        *slot.borrow_mut() = Some(CompletedStep {
            u: Vec::new(),
            v: Vec::new(),
            sab_len: 1,
            elapsed_ms,
        });
    }
}

fn build_blob_worker() -> Result<web_sys::Worker, JsValue> {
    let parts = js_sys::Array::of1(&JsValue::from_str(WORKER_JS_SOURCE));
    let bag = web_sys::BlobPropertyBag::new();
    bag.set_type("application/javascript");
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &bag)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)?;
    let opts = web_sys::WorkerOptions::new();
    opts.set_type(web_sys::WorkerType::Classic);
    let worker_res = web_sys::Worker::new_with_options(&url, &opts);
    let _ = web_sys::Url::revoke_object_url(&url);
    worker_res
}

fn build_job_object(kind: &str, job: VolumetricStepJob) -> js_sys::Object {
    let msg = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&msg, &JsValue::from_str("kind"), &JsValue::from_str(kind));
    for (k, val) in [
        ("dim", job.dim as f64),
        ("dt", job.dt as f64),
        ("du", job.du as f64),
        ("dv", job.dv as f64),
        ("a", job.a as f64),
        ("b", job.b as f64),
        ("substeps", job.substeps as f64),
    ] {
        let _ = js_sys::Reflect::set(&msg, &JsValue::from_str(k), &JsValue::from_f64(val));
    }
    msg
}
