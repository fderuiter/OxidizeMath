//! Asynchronous desktop and incremental WebAssembly GIF export jobs for [`super::SimulationFramework`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

use eframe::egui;
use egui_plot::export::{export_custom_image_with_bounds, ExportOptions, StreamingGifEncoder};

use super::export_figure::{
    render_progressive_primitive_slice, sample_tool_gif_payload, GifJobPayload,
};
use super::{ActionBarExportStatus, ExportFormat, ExportStatusSeverity, SimulationFramework};

struct GifFrameCursor {
    payload: GifJobPayload,
    options: ExportOptions,
    next_index: usize,
    total_frames: usize,
}

impl GifFrameCursor {
    fn new(payload: GifJobPayload, options: ExportOptions) -> Self {
        let total_frames = match &payload {
            GifJobPayload::Raster { grids, .. } => grids.len(),
            GifJobPayload::Datasets { frames, .. } => frames.len(),
            GifJobPayload::ProgressivePrimitives { total_steps, .. } => *total_steps,
        };
        Self {
            payload,
            options,
            next_index: 0,
            total_frames,
        }
    }

    fn is_finished(&self) -> bool {
        self.next_index >= self.total_frames
    }

    fn render_next_frame(&mut self) -> Option<Result<egui::ColorImage, String>> {
        if self.is_finished() {
            return None;
        }
        let idx = self.next_index;
        self.next_index += 1;
        let res = match &mut self.payload {
            GifJobPayload::Raster {
                grids,
                target_w,
                target_h,
            } => {
                let grid = std::mem::take(&mut grids[idx]);
                Ok(grid.into_scaled_color_image(*target_w, *target_h, &self.options))
            }
            GifJobPayload::Datasets { frames, bounds } => {
                let frame = std::mem::take(&mut frames[idx]);
                export_custom_image_with_bounds(&frame, &self.options, *bounds)
            }
            GifJobPayload::ProgressivePrimitives {
                primitives,
                bounds,
                total_steps,
            } => render_progressive_primitive_slice(
                primitives,
                *bounds,
                idx + 1,
                *total_steps,
                &self.options,
            ),
        };
        Some(res)
    }
}

#[cfg(not(target_arch = "wasm32"))]
enum GifWorkerEvent {
    Progress { completed: usize, total: usize },
    Finished(Result<Vec<u8>, String>),
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) struct FrameworkGifJob {
    filename: String,
    rx: mpsc::Receiver<GifWorkerEvent>,
    cancel: Arc<AtomicBool>,
    completed: usize,
    total: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for FrameworkGifJob {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_framework_gif_job(
    filename: String,
    cursor: GifFrameCursor,
    fps: u32,
    ctx: Option<&egui::Context>,
) -> Result<FrameworkGifJob, String> {
    let total = cursor.total_frames;
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = Arc::clone(&cancel);
    let worker_ctx = ctx.cloned();
    std::thread::spawn(move || {
        run_desktop_gif_worker(cursor, fps, &worker_cancel, &tx, worker_ctx.as_ref());
    });
    Ok(FrameworkGifJob {
        filename,
        rx,
        cancel,
        completed: 0,
        total,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn run_desktop_gif_worker(
    mut cursor: GifFrameCursor,
    fps: u32,
    cancel: &AtomicBool,
    tx: &mpsc::Sender<GifWorkerEvent>,
    ctx: Option<&egui::Context>,
) {
    let mut encoder = match StreamingGifEncoder::new(fps) {
        Ok(enc) => enc,
        Err(err) => {
            let _ = tx.send(GifWorkerEvent::Finished(Err(err)));
            if let Some(c) = ctx {
                c.request_repaint();
            }
            return;
        }
    };
    let total = cursor.total_frames;
    while !cancel.load(Ordering::Relaxed) {
        let Some(frame_res) = cursor.render_next_frame() else {
            break;
        };
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        let step_res = frame_res.and_then(|img| encoder.encode_color_image(img).map(|_| ()));
        if let Err(err) = step_res {
            let _ = tx.send(GifWorkerEvent::Finished(Err(err)));
            if let Some(c) = ctx {
                c.request_repaint();
            }
            return;
        }
        let _ = tx.send(GifWorkerEvent::Progress {
            completed: cursor.next_index,
            total,
        });
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }
    if cancel.load(Ordering::Relaxed) {
        return;
    }
    let _ = tx.send(GifWorkerEvent::Finished(encoder.finish()));
    if let Some(c) = ctx {
        c.request_repaint();
    }
}

#[cfg(target_arch = "wasm32")]
pub(super) struct FrameworkGifJob {
    filename: String,
    cursor: GifFrameCursor,
    encoder: Option<StreamingGifEncoder>,
    started_at: crate::web_io::PlatformInstant,
    completed: usize,
    total: usize,
}

#[cfg(target_arch = "wasm32")]
fn spawn_framework_gif_job(
    filename: String,
    cursor: GifFrameCursor,
    fps: u32,
    _ctx: Option<&egui::Context>,
) -> Result<FrameworkGifJob, String> {
    let total = cursor.total_frames;
    let encoder = StreamingGifEncoder::new(fps)?;
    Ok(FrameworkGifJob {
        filename,
        cursor,
        encoder: Some(encoder),
        started_at: crate::web_io::PlatformInstant::now(),
        completed: 0,
        total,
    })
}

impl SimulationFramework {
    /// Returns `(completed_frames, total_frames)` while an asynchronous or incremental GIF export is active.
    pub(crate) fn active_gif_progress(&self) -> Option<(usize, usize)> {
        self.gif_job.as_ref().map(|job| (job.completed, job.total))
    }

    /// Aborts any active GIF export job, freeing buffers and preventing stale downloads or status banners.
    pub(crate) fn cancel_active_gif_export(&mut self, user_initiated: bool) -> bool {
        self.export_queue.remove_format(ExportFormat::Gif);
        let Some(job) = self.gif_job.take() else {
            return false;
        };
        drop(job);
        if user_initiated {
            self.set_export_status(ExportStatusSeverity::Warning, "GIF export canceled.");
        } else {
            self.export_status_message = None;
            self.export_queue.status = None;
        }
        true
    }

    pub(super) fn start_gif_export(&mut self, ctx: Option<&egui::Context>) {
        if self.gif_job.is_some() {
            return;
        }
        let Some(tool_name) = self.ensure_figure_exportable() else {
            return;
        };
        let filename = Self::build_export_filename(&tool_name, ExportFormat::Gif);
        let fallback = self.export_datasets_with_fallback();
        let (frames, fps, opts) = (
            self.export_gif_frames as usize,
            self.export_gif_fps,
            self.export_options.clone(),
        );
        let tool = self
            .active_tool
            .as_mut()
            .expect("verified active tool")
            .as_mut();
        let payload = match sample_tool_gif_payload(tool, &fallback, &opts, frames) {
            Ok(payload) => payload,
            Err(err) => {
                self.set_export_status(
                    ExportStatusSeverity::Error,
                    format!("GIF export failed: {err}"),
                );
                return;
            }
        };
        let cursor = GifFrameCursor::new(payload, opts);
        match spawn_framework_gif_job(filename, cursor, fps, ctx) {
            Ok(job) => {
                let total = job.total;
                self.gif_job = Some(job);
                self.set_gif_progress_status(0, total, true);
                if let Some(c) = ctx {
                    c.request_repaint();
                }
            }
            Err(err) => {
                self.set_export_status(
                    ExportStatusSeverity::Error,
                    format!("GIF export failed: {err}"),
                );
            }
        }
    }

    fn set_gif_progress_status(&mut self, completed: usize, total: usize, announce: bool) {
        let msg = format!("Encoding GIF frame {completed}/{total}...");
        if announce {
            crate::accessibility::announce_status_with_priority(
                &msg,
                ExportStatusSeverity::Success.aria_priority(),
            );
        }
        self.export_status_severity = ExportStatusSeverity::Success;
        self.export_status_message = Some(msg.clone());
        self.export_queue.status = Some(ActionBarExportStatus {
            message: msg,
            is_error: false,
            expires_at_time: None,
        });
    }

    pub(crate) fn poll_active_gif_export(&mut self, ctx: &egui::Context) {
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_desktop_gif_export(ctx);
        #[cfg(target_arch = "wasm32")]
        self.step_wasm_gif_export(ctx);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn poll_desktop_gif_export(&mut self, ctx: &egui::Context) {
        let Some(job) = self.gif_job.as_mut() else {
            return;
        };
        let mut progress_update = None;
        let mut finished = None;
        loop {
            match job.rx.try_recv() {
                Ok(GifWorkerEvent::Progress { completed, total }) => {
                    job.completed = completed;
                    job.total = total;
                    progress_update = Some((completed, total));
                }
                Ok(GifWorkerEvent::Finished(res)) => {
                    finished = Some(res);
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    finished = Some(Err("GIF worker disconnected unexpectedly".to_string()));
                    break;
                }
            }
        }
        if let Some(result) = finished {
            let done_job = self.gif_job.take().expect("active GIF job");
            self.finish_gif_export_result(&done_job.filename, result);
            ctx.request_repaint();
            return;
        }
        if let Some((completed, total)) = progress_update {
            self.set_gif_progress_status(completed, total, false);
        }
        ctx.request_repaint();
    }

    #[cfg(target_arch = "wasm32")]
    fn step_wasm_gif_export(&mut self, ctx: &egui::Context) {
        let Some(job) = self.gif_job.as_mut() else {
            return;
        };
        let _elapsed = job.started_at.elapsed();
        if let Some(frame_res) = job.cursor.render_next_frame() {
            let step_err = match (frame_res, job.encoder.as_mut()) {
                (Ok(img), Some(enc)) => enc.encode_color_image(img).err(),
                (Err(err), _) => Some(err),
                (Ok(_), None) => Some("GIF encoder already finalized".to_string()),
            };
            if let Some(err) = step_err {
                let done_job = self.gif_job.take().expect("active GIF job");
                self.finish_gif_export_result(&done_job.filename, Err(err));
                ctx.request_repaint();
                return;
            }
            job.completed = job.cursor.next_index;
        }
        if job.cursor.is_finished() {
            let mut done_job = self.gif_job.take().expect("active GIF job");
            let res = done_job
                .encoder
                .take()
                .ok_or_else(|| "GIF encoder missing".to_string())
                .and_then(StreamingGifEncoder::finish);
            self.finish_gif_export_result(&done_job.filename, res);
            ctx.request_repaint();
            return;
        }
        let (completed, total) = (job.completed, job.total);
        self.set_gif_progress_status(completed, total, false);
        ctx.request_repaint();
    }

    fn finish_gif_export_result(&mut self, filename: &str, result: Result<Vec<u8>, String>) {
        match result {
            Ok(bytes) => self.complete_file_export(
                ExportFormat::Gif,
                filename,
                &bytes,
                None,
                "GIF animation",
                "GIF export failed",
            ),
            Err(err) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("GIF export failed: {err}"),
            ),
        }
    }
}

#[cfg(test)]
#[path = "export_gif_job_tests.rs"]
mod tests;
