//! Thin `wasm-bindgen` / `web-sys` WebGPU (`navigator.gpu`) bindings, `GpuComputePipeline`
//! cache, and double-buffered `MAP_READ` `GpuBuffer` staging ring for `wasm32-unknown-unknown`.

use super::{
    WebGpuDispatchSpec, WebGpuDispatcherState, GPU_BUFFER_USAGE_COPY_DST,
    GPU_BUFFER_USAGE_COPY_SRC, GPU_BUFFER_USAGE_MAP_READ, GPU_BUFFER_USAGE_STORAGE,
    GPU_BUFFER_USAGE_UNIFORM, GPU_MAP_MODE_READ,
};
use crate::web_io::PlatformInstant;
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use wasm_bindgen::{prelude::*, JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = Object)]
    #[derive(Clone)]
    type Gpu;
    #[wasm_bindgen(method, catch, js_name = requestAdapter)]
    fn request_adapter(this: &Gpu) -> Result<js_sys::Promise, JsValue>;

    #[wasm_bindgen(js_name = Object)]
    #[derive(Clone)]
    type GpuAdapter;
    #[wasm_bindgen(method, catch, js_name = requestDevice)]
    fn request_device(this: &GpuAdapter) -> Result<js_sys::Promise, JsValue>;

    #[wasm_bindgen(js_name = Object)]
    #[derive(Clone)]
    type GpuDevice;
    #[wasm_bindgen(method, getter)]
    fn queue(this: &GpuDevice) -> GpuQueue;
    #[wasm_bindgen(method, catch, js_name = createShaderModule)]
    fn create_shader_module(this: &GpuDevice, desc: &JsValue) -> Result<GpuShaderModule, JsValue>;
    #[wasm_bindgen(method, catch, js_name = createComputePipeline)]
    fn create_compute_pipeline(
        this: &GpuDevice,
        desc: &JsValue,
    ) -> Result<GpuComputePipeline, JsValue>;
    #[wasm_bindgen(method, catch, js_name = createBuffer)]
    fn create_buffer(this: &GpuDevice, desc: &JsValue) -> Result<GpuBuffer, JsValue>;
    #[wasm_bindgen(method, catch, js_name = createBindGroup)]
    fn create_bind_group(this: &GpuDevice, desc: &JsValue) -> Result<GpuBindGroup, JsValue>;
    #[wasm_bindgen(method, catch, js_name = createCommandEncoder)]
    fn create_command_encoder(this: &GpuDevice) -> Result<GpuCommandEncoder, JsValue>;

    #[wasm_bindgen(js_name = Object)]
    #[derive(Clone)]
    type GpuQueue;
    #[wasm_bindgen(method, catch, js_name = writeBuffer)]
    fn write_buffer(
        this: &GpuQueue,
        buffer: &GpuBuffer,
        offset: f64,
        data: &JsValue,
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(method, catch)]
    fn submit(this: &GpuQueue, command_buffers: &js_sys::Array) -> Result<(), JsValue>;

    #[wasm_bindgen(js_name = Object)]
    type GpuShaderModule;

    #[wasm_bindgen(js_name = Object)]
    #[derive(Clone)]
    type GpuComputePipeline;
    #[wasm_bindgen(method, catch, js_name = getBindGroupLayout)]
    fn get_bind_group_layout(this: &GpuComputePipeline, index: u32) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(js_name = Object)]
    #[derive(Clone)]
    type GpuBuffer;
    #[wasm_bindgen(method, catch, js_name = mapAsync)]
    fn map_async(this: &GpuBuffer, mode: u32) -> Result<js_sys::Promise, JsValue>;
    #[wasm_bindgen(method, catch, js_name = getMappedRange)]
    fn get_mapped_range(this: &GpuBuffer) -> Result<js_sys::ArrayBuffer, JsValue>;
    #[wasm_bindgen(method, catch)]
    fn unmap(this: &GpuBuffer) -> Result<(), JsValue>;
    #[wasm_bindgen(method, catch)]
    fn destroy(this: &GpuBuffer) -> Result<(), JsValue>;

    #[wasm_bindgen(js_name = Object)]
    type GpuBindGroup;

    #[wasm_bindgen(js_name = Object)]
    type GpuCommandEncoder;
    #[wasm_bindgen(method, catch, js_name = beginComputePass)]
    fn begin_compute_pass(this: &GpuCommandEncoder) -> Result<GpuComputePassEncoder, JsValue>;
    #[wasm_bindgen(method, catch, js_name = copyBufferToBuffer)]
    fn copy_buffer_to_buffer(
        this: &GpuCommandEncoder,
        src: &GpuBuffer,
        src_offset: f64,
        dst: &GpuBuffer,
        dst_offset: f64,
        size: f64,
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(method, catch)]
    fn finish(this: &GpuCommandEncoder) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(js_name = Object)]
    type GpuComputePassEncoder;
    #[wasm_bindgen(method, catch, js_name = setPipeline)]
    fn set_pipeline(
        this: &GpuComputePassEncoder,
        pipeline: &GpuComputePipeline,
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(method, catch, js_name = setBindGroup)]
    fn set_bind_group(
        this: &GpuComputePassEncoder,
        index: u32,
        bind_group: &GpuBindGroup,
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(method, catch, js_name = dispatchWorkgroups)]
    fn dispatch_workgroups(
        this: &GpuComputePassEncoder,
        x: u32,
        y: u32,
        z: u32,
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(method, catch)]
    fn end(this: &GpuComputePassEncoder) -> Result<(), JsValue>;
}

struct CachedComputeBuffers {
    uniform_buf: GpuBuffer,
    uniform_cap: usize,
    input_buf: GpuBuffer,
    input_cap: usize,
    output_buf: GpuBuffer,
    staging_pair: [GpuBuffer; 2],
    output_bytes: usize,
}

struct ReadyGpuContext {
    _adapter: GpuAdapter,
    device: GpuDevice,
    pipeline_cache: Option<(u64, GpuComputePipeline)>,
    buffers: Option<CachedComputeBuffers>,
}

struct ReadbackPayload {
    values: Vec<f32>,
    elapsed_ms: f64,
}

struct SharedDispatcherSlot {
    state: WebGpuDispatcherState,
    ctx: Option<ReadyGpuContext>,
    staging_in_flight: [bool; 2],
    completed: Option<ReadbackPayload>,
}

pub(super) struct WasmWebGpuDispatcher {
    shared: Rc<RefCell<SharedDispatcherSlot>>,
    staging_slot: usize,
}

fn lookup_navigator_gpu() -> Option<Gpu> {
    let win = js_sys::global().dyn_into::<web_sys::Window>().ok()?;
    let gpu = js_sys::Reflect::get(&win.navigator(), &JsValue::from_str("gpu")).ok()?;
    if gpu.is_undefined() || gpu.is_null() {
        return None;
    }
    Some(gpu.unchecked_into::<Gpu>())
}

pub(super) fn has_navigator_gpu() -> bool {
    lookup_navigator_gpu().is_some()
}

fn set_prop(obj: &js_sys::Object, key: &str, val: &JsValue) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), val);
}

fn alloc_gpu_buffer(device: &GpuDevice, byte_size: usize, usage: u32) -> Option<GpuBuffer> {
    let aligned = byte_size.max(16).next_multiple_of(16);
    let desc = js_sys::Object::new();
    set_prop(&desc, "size", &JsValue::from_f64(aligned as f64));
    set_prop(&desc, "usage", &JsValue::from_f64(f64::from(usage)));
    device.create_buffer(&desc).ok()
}

fn build_compute_pipeline(
    device: &GpuDevice,
    label: &str,
    wgsl: &str,
    entry_point: &str,
) -> Option<GpuComputePipeline> {
    let mod_desc = js_sys::Object::new();
    set_prop(&mod_desc, "label", &JsValue::from_str(label));
    set_prop(&mod_desc, "code", &JsValue::from_str(wgsl));
    let shader_mod = device.create_shader_module(&mod_desc).ok()?;

    let stage = js_sys::Object::new();
    set_prop(&stage, "module", shader_mod.as_ref());
    set_prop(&stage, "entryPoint", &JsValue::from_str(entry_point));

    let pipe_desc = js_sys::Object::new();
    set_prop(&pipe_desc, "label", &JsValue::from_str(label));
    set_prop(&pipe_desc, "layout", &JsValue::from_str("auto"));
    set_prop(&pipe_desc, "compute", &stage);
    device.create_compute_pipeline(&pipe_desc).ok()
}

fn build_three_binding_group(
    device: &GpuDevice,
    pipeline: &GpuComputePipeline,
    bufs: &CachedComputeBuffers,
) -> Option<GpuBindGroup> {
    let layout = pipeline.get_bind_group_layout(0).ok()?;
    let entries = js_sys::Array::new();
    for (idx, buf) in [
        (0u32, &bufs.uniform_buf),
        (1u32, &bufs.input_buf),
        (2u32, &bufs.output_buf),
    ] {
        let res = js_sys::Object::new();
        set_prop(&res, "buffer", buf.as_ref());
        let entry = js_sys::Object::new();
        set_prop(&entry, "binding", &JsValue::from_f64(f64::from(idx)));
        set_prop(&entry, "resource", &res);
        entries.push(&entry);
    }
    let desc = js_sys::Object::new();
    set_prop(&desc, "layout", &layout);
    set_prop(&desc, "entries", &entries);
    device.create_bind_group(&desc).ok()
}

fn ensure_buffers<'a>(
    device: &GpuDevice,
    slot: &'a mut Option<CachedComputeBuffers>,
    u_bytes: usize,
    in_bytes: usize,
    out_bytes: usize,
) -> Option<&'a CachedComputeBuffers> {
    let needs_realloc = slot.as_ref().is_none_or(|b| {
        b.uniform_cap < u_bytes || b.input_cap < in_bytes || b.output_bytes != out_bytes
    });
    if needs_realloc {
        if let Some(old) = slot.take() {
            let _ = old.uniform_buf.destroy();
            let _ = old.input_buf.destroy();
            let _ = old.output_buf.destroy();
            let _ = old.staging_pair[0].destroy();
            let _ = old.staging_pair[1].destroy();
        }
        let u_cap = u_bytes.max(16).next_multiple_of(16);
        let in_cap = in_bytes.max(16).next_multiple_of(16);
        let out_cap = out_bytes.max(16).next_multiple_of(16);
        let st_usage = GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST;
        *slot = Some(CachedComputeBuffers {
            uniform_buf: alloc_gpu_buffer(
                device,
                u_cap,
                GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
            )?,
            uniform_cap: u_cap,
            input_buf: alloc_gpu_buffer(
                device,
                in_cap,
                GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
            )?,
            input_cap: in_cap,
            output_buf: alloc_gpu_buffer(
                device,
                out_cap,
                GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC,
            )?,
            staging_pair: [
                alloc_gpu_buffer(device, out_cap, st_usage)?,
                alloc_gpu_buffer(device, out_cap, st_usage)?,
            ],
            output_bytes: out_bytes,
        });
    }
    slot.as_ref()
}

impl WasmWebGpuDispatcher {
    pub(super) fn new() -> Self {
        let initial_state = if has_navigator_gpu() {
            WebGpuDispatcherState::Uninitialized
        } else {
            WebGpuDispatcherState::Unavailable
        };
        Self {
            shared: Rc::new(RefCell::new(SharedDispatcherSlot {
                state: initial_state,
                ctx: None,
                staging_in_flight: [false, false],
                completed: None,
            })),
            staging_slot: 0,
        }
    }

    pub(super) fn state(&self) -> WebGpuDispatcherState {
        self.shared.borrow().state
    }

    pub(super) const fn active_staging_slot(&self) -> usize {
        self.staging_slot
    }

    fn start_async_init(&self) {
        let Some(gpu) = lookup_navigator_gpu() else {
            self.shared.borrow_mut().state = WebGpuDispatcherState::Unavailable;
            return;
        };
        self.shared.borrow_mut().state = WebGpuDispatcherState::Initializing;
        let shared = Rc::clone(&self.shared);
        wasm_bindgen_futures::spawn_local(async move {
            let adapter_val = match gpu.request_adapter() {
                Ok(p) => JsFuture::from(p).await.ok(),
                Err(_) => None,
            };
            let Some(adapter_js) = adapter_val.filter(|v| !v.is_null() && !v.is_undefined()) else {
                shared.borrow_mut().state = WebGpuDispatcherState::Unavailable;
                return;
            };
            let adapter: GpuAdapter = adapter_js.unchecked_into();
            let device_val = match adapter.request_device() {
                Ok(p) => JsFuture::from(p).await.ok(),
                Err(_) => None,
            };
            let Some(device_js) = device_val.filter(|v| !v.is_null() && !v.is_undefined()) else {
                shared.borrow_mut().state = WebGpuDispatcherState::Unavailable;
                return;
            };
            let mut guard = shared.borrow_mut();
            guard.ctx = Some(ReadyGpuContext {
                _adapter: adapter,
                device: device_js.unchecked_into(),
                pipeline_cache: None,
                buffers: None,
            });
            guard.state = WebGpuDispatcherState::Ready;
        });
    }

    pub(super) fn submit_compute(&mut self, spec: WebGpuDispatchSpec<'_>) -> bool {
        let state = self.state();
        if state == WebGpuDispatcherState::Uninitialized {
            self.start_async_init();
            return false;
        }
        if state != WebGpuDispatcherState::Ready {
            return false;
        }
        let slot_idx = self.staging_slot;
        if self.shared.borrow().staging_in_flight[slot_idx] {
            return false;
        }
        let Some((staging_buf, out_len)) = self.encode_and_submit(spec, slot_idx) else {
            return false;
        };
        self.staging_slot ^= 1;
        self.spawn_staging_readback(staging_buf, slot_idx, out_len);
        true
    }

    fn encode_and_submit(
        &self,
        spec: WebGpuDispatchSpec<'_>,
        slot_idx: usize,
    ) -> Option<(GpuBuffer, usize)> {
        let mut guard = self.shared.borrow_mut();
        let ctx = guard.ctx.as_mut()?;
        let mut hasher = DefaultHasher::new();
        spec.wgsl_source.hash(&mut hasher);
        spec.entry_point.hash(&mut hasher);
        let key = hasher.finish();
        if ctx.pipeline_cache.as_ref().is_none_or(|(k, _)| *k != key) {
            let p = build_compute_pipeline(
                &ctx.device,
                spec.shader_label,
                spec.wgsl_source,
                spec.entry_point,
            )?;
            ctx.pipeline_cache = Some((key, p));
        }
        let pipeline = ctx.pipeline_cache.as_ref()?.1.clone();
        let out_bytes = spec.output_len.checked_mul(4)?;
        let bufs = ensure_buffers(
            &ctx.device,
            &mut ctx.buffers,
            spec.uniform_bytes.len(),
            spec.input_f32.len() * 4,
            out_bytes,
        )?;
        let queue = ctx.device.queue();
        if !spec.uniform_bytes.is_empty() {
            let u_arr = js_sys::Uint8Array::from(spec.uniform_bytes);
            queue.write_buffer(&bufs.uniform_buf, 0.0, &u_arr).ok()?;
        }
        if !spec.input_f32.is_empty() {
            let in_arr = js_sys::Float32Array::from(spec.input_f32);
            queue.write_buffer(&bufs.input_buf, 0.0, &in_arr).ok()?;
        }
        let bg = build_three_binding_group(&ctx.device, &pipeline, bufs)?;
        let enc = ctx.device.create_command_encoder().ok()?;
        let pass = enc.begin_compute_pass().ok()?;
        pass.set_pipeline(&pipeline).ok()?;
        pass.set_bind_group(0, &bg).ok()?;
        let [wx, wy, wz] = spec.workgroups;
        pass.dispatch_workgroups(wx.max(1), wy.max(1), wz.max(1))
            .ok()?;
        pass.end().ok()?;
        let staging = bufs.staging_pair[slot_idx].clone();
        enc.copy_buffer_to_buffer(&bufs.output_buf, 0.0, &staging, 0.0, out_bytes as f64)
            .ok()?;
        let cmd_buf = enc.finish().ok()?;
        let cmds = js_sys::Array::of1(&cmd_buf);
        queue.submit(&cmds).ok()?;
        guard.staging_in_flight[slot_idx] = true;
        Some((staging, spec.output_len))
    }

    fn spawn_staging_readback(&self, staging: GpuBuffer, slot_idx: usize, out_len: usize) {
        let shared = Rc::clone(&self.shared);
        let t0 = PlatformInstant::now();
        wasm_bindgen_futures::spawn_local(async move {
            let mapped_ok = match staging.map_async(GPU_MAP_MODE_READ) {
                Ok(p) => JsFuture::from(p).await.is_ok(),
                Err(_) => false,
            };
            let mut payload = None;
            if mapped_ok {
                if let Ok(range) = staging.get_mapped_range() {
                    let view = js_sys::Float32Array::new(&range);
                    let copy_len = (view.length() as usize).min(out_len);
                    let mut vals = vec![0.0_f32; copy_len];
                    view.subarray(0, copy_len as u32).copy_to(&mut vals);
                    payload = Some(ReadbackPayload {
                        values: vals,
                        elapsed_ms: (t0.elapsed().as_secs_f64() * 1_000.0).max(0.05),
                    });
                }
                let _ = staging.unmap();
            }
            let mut guard = shared.borrow_mut();
            guard.staging_in_flight[slot_idx] = false;
            if let Some(done) = payload {
                guard.completed = Some(done);
            }
        });
    }

    pub(super) fn poll_readback(&mut self, dst: &mut [f32]) -> Option<f64> {
        let done = self.shared.borrow_mut().completed.take()?;
        let n = done.values.len().min(dst.len());
        dst[..n].copy_from_slice(&done.values[..n]);
        Some(done.elapsed_ms)
    }
}
