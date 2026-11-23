//! Non-blocking standalone animated GIF export job tracking for [`super::ExportModalState`].

#[cfg(target_arch = "wasm32")]
use std::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use crate::export::{
    DatasetFrame, ExportOptions, StreamingGifEncoder, compute_dataset_frames_bounds,
    export_custom_image_with_bounds, save_file_dialog,
};
use crate::export_primitives::resolve_canvas_dimensions;

#[cfg(not(target_arch = "wasm32"))]
enum ModalGifEvent {
    Progress { completed: usize, total: usize },
    Finished(Result<Vec<u8>, String>),
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeModalGifJob {
    rx: mpsc::Receiver<ModalGifEvent>,
    cancel: Arc<AtomicBool>,
    completed: usize,
    total: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for NativeModalGifJob {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

#[cfg(target_arch = "wasm32")]
struct WasmModalGifJob {
    frames: VecDeque<DatasetFrame>,
    custom_bounds: Option<[f64; 4]>,
    options: ExportOptions,
    encoder: Option<StreamingGifEncoder>,
    completed: usize,
    total: usize,
}

enum ModalGifJobInner {
    #[cfg(not(target_arch = "wasm32"))]
    Native(NativeModalGifJob),
    #[cfg(target_arch = "wasm32")]
    Wasm(WasmModalGifJob),
}

/// Shared handle tracking an in-flight standalone GIF export job inside [`super::ExportModalState`].
///
/// Cloning shares the underlying job state across `egui` frame snapshots, while `serde`
/// skips transient background job handles cleanly.
#[derive(Clone, Default)]
pub struct ModalGifJobHandle {
    inner: Option<Arc<Mutex<ModalGifJobInner>>>,
}

impl std::fmt::Debug for ModalGifJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModalGifJobHandle")
            .field("active", &self.is_active())
            .field("progress", &self.progress())
            .finish()
    }
}

impl PartialEq for ModalGifJobHandle {
    fn eq(&self, other: &Self) -> bool {
        match (&self.inner, &other.inner) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl ModalGifJobHandle {
    /// Return `true` while a standalone GIF export job is actively encoding frames.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.inner.is_some()
    }

    /// Return `(completed_frames, total_frames)` when a standalone GIF export is active.
    #[must_use]
    pub fn progress(&self) -> Option<(usize, usize)> {
        let arc = self.inner.as_ref()?;
        let guard = arc.lock().unwrap_or_else(|p| p.into_inner());
        match &*guard {
            #[cfg(not(target_arch = "wasm32"))]
            ModalGifJobInner::Native(job) => Some((job.completed, job.total)),
            #[cfg(target_arch = "wasm32")]
            ModalGifJobInner::Wasm(job) => Some((job.completed, job.total)),
        }
    }

    /// Abort any active standalone GIF export job and release buffered inputs immediately.
    pub fn cancel(&mut self) -> bool {
        let Some(arc) = self.inner.take() else {
            return false;
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            let guard = arc.lock().unwrap_or_else(|p| p.into_inner());
            let ModalGifJobInner::Native(job) = &*guard;
            job.cancel.store(true, Ordering::Relaxed);
        }
        #[cfg(target_arch = "wasm32")]
        let _ = arc;
        true
    }

    pub(crate) fn start(
        &mut self,
        ctx: &egui::Context,
        frames: Vec<DatasetFrame>,
        options: &ExportOptions,
        fps: u32,
        status_message: &mut Option<String>,
    ) {
        if self.is_active() {
            return;
        }
        if let Err(err) = validate_standalone_gif_inputs(&frames, options) {
            let msg = format!("GIF export failed: {err}");
            crate::emit_plot_live_message(ctx, msg.clone(), true);
            *status_message = Some(msg);
            return;
        }
        let total = frames.len();
        let bounds = compute_dataset_frames_bounds(&frames, options);
        match build_job_inner(ctx, frames, bounds, options.clone(), fps, total) {
            Ok(inner) => {
                self.inner = Some(Arc::new(Mutex::new(inner)));
                let msg = format!("Encoding GIF frame 0/{total}...");
                crate::emit_plot_live_message(ctx, msg.clone(), false);
                *status_message = Some(msg);
                ctx.request_repaint();
            }
            Err(err) => {
                let msg = format!("GIF export failed: {err}");
                crate::emit_plot_live_message(ctx, msg.clone(), true);
                *status_message = Some(msg);
            }
        }
    }

    pub(crate) fn poll(
        &mut self,
        ctx: &egui::Context,
        filename: &str,
        status_message: &mut Option<String>,
    ) {
        let Some(arc) = self.inner.clone() else {
            return;
        };
        let finished = {
            let mut guard = arc.lock().unwrap_or_else(|p| p.into_inner());
            match &mut *guard {
                #[cfg(not(target_arch = "wasm32"))]
                ModalGifJobInner::Native(job) => poll_native_job(ctx, job, status_message),
                #[cfg(target_arch = "wasm32")]
                ModalGifJobInner::Wasm(job) => step_wasm_job(ctx, job, status_message),
            }
        };
        if let Some(result) = finished {
            self.inner = None;
            finalize_modal_gif(ctx, filename, result, status_message);
        }
    }
}

fn validate_standalone_gif_inputs(
    frames: &[DatasetFrame],
    options: &ExportOptions,
) -> Result<(), String> {
    if frames.is_empty() {
        return Err("No dataset frames provided for GIF export".to_string());
    }
    resolve_canvas_dimensions(options).map(|_| ())
}

fn finalize_modal_gif(
    ctx: &egui::Context,
    filename: &str,
    result: Result<Vec<u8>, String>,
    status_message: &mut Option<String>,
) {
    let (msg, assertive) = match result {
        Ok(gif_bytes) => {
            save_file_dialog(filename, &gif_bytes, "GIF Animation", &["gif"]);
            ("GIF animation saved successfully.".to_string(), false)
        }
        Err(err) => (format!("GIF export failed: {err}"), true),
    };
    crate::emit_plot_live_message(ctx, msg.clone(), assertive);
    *status_message = Some(msg);
    ctx.request_repaint();
}

#[cfg(not(target_arch = "wasm32"))]
struct StandaloneGifTask {
    frames: Vec<DatasetFrame>,
    bounds: Option<[f64; 4]>,
    options: ExportOptions,
    fps: u32,
    total: usize,
}

#[cfg(not(target_arch = "wasm32"))]
fn build_job_inner(
    ctx: &egui::Context,
    frames: Vec<DatasetFrame>,
    bounds: Option<[f64; 4]>,
    options: ExportOptions,
    fps: u32,
    total: usize,
) -> Result<ModalGifJobInner, String> {
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = Arc::clone(&cancel);
    let worker_ctx = ctx.clone();
    let task = StandaloneGifTask {
        frames,
        bounds,
        options,
        fps,
        total,
    };
    std::thread::spawn(move || {
        run_native_modal_gif_worker(task, &worker_cancel, &tx, &worker_ctx);
    });
    Ok(ModalGifJobInner::Native(NativeModalGifJob {
        rx,
        cancel,
        completed: 0,
        total,
    }))
}

#[cfg(not(target_arch = "wasm32"))]
fn run_native_modal_gif_worker(
    task: StandaloneGifTask,
    cancel: &AtomicBool,
    tx: &mpsc::Sender<ModalGifEvent>,
    ctx: &egui::Context,
) {
    let mut encoder = match StreamingGifEncoder::new(task.fps) {
        Ok(enc) => enc,
        Err(err) => {
            let _ = tx.send(ModalGifEvent::Finished(Err(err)));
            ctx.request_repaint();
            return;
        }
    };
    for (idx, frame) in task.frames.into_iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        let img = match export_custom_image_with_bounds(&frame, &task.options, task.bounds) {
            Ok(img) => img,
            Err(err) => {
                let _ = tx.send(ModalGifEvent::Finished(Err(err)));
                ctx.request_repaint();
                return;
            }
        };
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        if let Err(err) = encoder.encode_color_image(img) {
            let _ = tx.send(ModalGifEvent::Finished(Err(err)));
            ctx.request_repaint();
            return;
        }
        let _ = tx.send(ModalGifEvent::Progress {
            completed: idx + 1,
            total: task.total,
        });
        ctx.request_repaint();
    }
    if cancel.load(Ordering::Relaxed) {
        return;
    }
    let _ = tx.send(ModalGifEvent::Finished(encoder.finish()));
    ctx.request_repaint();
}

#[cfg(not(target_arch = "wasm32"))]
fn poll_native_job(
    ctx: &egui::Context,
    job: &mut NativeModalGifJob,
    status_message: &mut Option<String>,
) -> Option<Result<Vec<u8>, String>> {
    let mut updated_progress = false;
    loop {
        match job.rx.try_recv() {
            Ok(ModalGifEvent::Progress { completed, total }) => {
                job.completed = completed;
                job.total = total;
                updated_progress = true;
            }
            Ok(ModalGifEvent::Finished(res)) => return Some(res),
            Err(mpsc::TryRecvError::Empty) => break,
            Err(mpsc::TryRecvError::Disconnected) => {
                return Some(Err("GIF worker disconnected unexpectedly".to_string()));
            }
        }
    }
    if updated_progress {
        *status_message = Some(format!(
            "Encoding GIF frame {}/{}...",
            job.completed, job.total
        ));
    }
    ctx.request_repaint();
    None
}

#[cfg(target_arch = "wasm32")]
fn build_job_inner(
    _ctx: &egui::Context,
    frames: Vec<DatasetFrame>,
    bounds: Option<[f64; 4]>,
    options: ExportOptions,
    fps: u32,
    total: usize,
) -> Result<ModalGifJobInner, String> {
    let encoder = StreamingGifEncoder::new(fps)?;
    Ok(ModalGifJobInner::Wasm(WasmModalGifJob {
        frames: frames.into_iter().collect(),
        custom_bounds: bounds,
        options,
        encoder: Some(encoder),
        completed: 0,
        total,
    }))
}

#[cfg(target_arch = "wasm32")]
fn step_wasm_job(
    ctx: &egui::Context,
    job: &mut WasmModalGifJob,
    status_message: &mut Option<String>,
) -> Option<Result<Vec<u8>, String>> {
    let Some(frame) = job.frames.pop_front() else {
        let enc = job
            .encoder
            .take()
            .ok_or_else(|| "GIF encoder missing".to_string());
        return Some(enc.and_then(StreamingGifEncoder::finish));
    };
    let img = match export_custom_image_with_bounds(&frame, &job.options, job.custom_bounds) {
        Ok(img) => img,
        Err(err) => return Some(Err(err)),
    };
    if let Some(enc) = job.encoder.as_mut() {
        if let Err(err) = enc.encode_color_image(img) {
            return Some(Err(err));
        }
    } else {
        return Some(Err("GIF encoder already finalized".to_string()));
    }
    job.completed += 1;
    if job.frames.is_empty() {
        let enc = job
            .encoder
            .take()
            .ok_or_else(|| "GIF encoder missing".to_string());
        return Some(enc.and_then(StreamingGifEncoder::finish));
    }
    *status_message = Some(format!(
        "Encoding GIF frame {}/{}...",
        job.completed, job.total
    ));
    ctx.request_repaint();
    None
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "wasm32")]
    use super::{ExportOptions, ModalGifJobHandle};
    use crate::{Line, Plot, PlotPoints};

    #[test]
    fn test_standalone_plot_show_save_gif_returns_without_blocking_inline() {
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let plot_id = egui::Id::new("standalone_plot_async_gif");
        let raw = || egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(960.0, 640.0),
            )),
            ..Default::default()
        };
        let step_plot = |input: egui::RawInput| {
            ctx.run(input, |c| {
                egui::CentralPanel::default().show(c, |ui| {
                    let pts: PlotPoints<'_> = (0..40)
                        .map(|i| [f64::from(i), (f64::from(i) * 0.2).sin()])
                        .collect();
                    Plot::new(plot_id).id(plot_id).show(ui, |plot_ui| {
                        plot_ui.line(Line::new("Wave", pts));
                    });
                });
            })
        };

        let _ = step_plot(raw());
        ctx.data_mut(|d| {
            let mut st = d
                .get_temp::<crate::AccessState>(plot_id)
                .unwrap_or_default();
            st.export_modal.open = true;
            st.export_modal.options.width = 640;
            st.export_modal.options.height = 480;
            st.export_modal.gif_frames = 12;
            d.insert_temp(plot_id, st);
        });
        let _ = step_plot(raw());
        let out_open = step_plot(raw());
        let save_pos = out_open
            .platform_output
            .accesskit_update
            .as_ref()
            .and_then(|u| {
                u.nodes.iter().find_map(|(_, n)| {
                    n.label()
                        .is_some_and(|l| l.contains("Save GIF"))
                        .then(|| n.bounds())
                        .flatten()
                        .map(|b| {
                            egui::pos2(((b.x0 + b.x1) * 0.5) as f32, ((b.y0 + b.y1) * 0.5) as f32)
                        })
                })
            })
            .expect("Save GIF button in standalone Plot::show modal");

        let click = |pressed: bool| {
            let mut r = raw();
            r.events = vec![
                egui::Event::PointerMoved(save_pos),
                egui::Event::PointerButton {
                    pos: save_pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ];
            r
        };
        let _ = step_plot(click(true));
        let _ = step_plot(click(false));
        let active_after_click = ctx
            .data(|d| d.get_temp::<crate::AccessState>(plot_id))
            .is_some_and(|st| st.export_modal.gif_job.is_active());
        assert!(
            active_after_click,
            "Standalone Plot::show must start async GIF job without blocking inline"
        );

        for _ in 0..250 {
            let still_active = ctx
                .data(|d| d.get_temp::<crate::AccessState>(plot_id))
                .is_some_and(|st| st.export_modal.gif_job.is_active());
            if !still_active {
                break;
            }
            #[cfg(not(target_arch = "wasm32"))]
            std::thread::sleep(std::time::Duration::from_millis(10));
            let _ = step_plot(raw());
        }
        let final_status = ctx
            .data(|d| d.get_temp::<crate::AccessState>(plot_id))
            .and_then(|st| st.export_modal.status_message);
        assert_eq!(
            final_status.as_deref(),
            Some("GIF animation saved successfully.")
        );
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    fn wasm_modal_gif_job_handle_steps_incrementally_and_cancels() {
        let ctx = egui::Context::default();
        let mut handle = ModalGifJobHandle::default();
        let mut status = None;
        let opts = ExportOptions {
            width: 120,
            height: 90,
            ..Default::default()
        };
        let frames = crate::export::sample_progressive_frames(
            &[("W".to_string(), vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])],
            3,
        );
        handle.start(&ctx, frames, &opts, 15, &mut status);
        assert_eq!(handle.progress(), Some((0, 3)));
        handle.poll(&ctx, "w.gif", &mut status);
        assert_eq!(handle.progress(), Some((1, 3)));
        assert!(handle.cancel());
        assert_eq!(handle.progress(), None);
    }
}
