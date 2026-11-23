//! Plot-scoped asynchronous CSV dataset and import status queues.

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock, PoisonError};

use crate::csv_import::{CsvImportError, CsvImportOutcome, parse_csv_bounded};
use crate::export::ImportedDataset;

/// Maximum number of unconsumed plot CSV import payloads or status messages retained globally.
pub(super) const MAX_PENDING_PLOT_IMPORTS: usize = 64;

pub(super) type PendingImport = (egui::Id, Vec<ImportedDataset>);
pub(super) type PendingImportStatus = (egui::Id, String, bool);

pub(super) fn pending_imports_queue() -> &'static Mutex<VecDeque<PendingImport>> {
    static QUEUE: OnceLock<Mutex<VecDeque<PendingImport>>> = OnceLock::new();
    QUEUE.get_or_init(|| Mutex::new(VecDeque::new()))
}

pub(super) fn pending_status_queue() -> &'static Mutex<VecDeque<PendingImportStatus>> {
    static QUEUE: OnceLock<Mutex<VecDeque<PendingImportStatus>>> = OnceLock::new();
    QUEUE.get_or_init(|| Mutex::new(VecDeque::new()))
}

fn push_bounded<T>(queue: &Mutex<VecDeque<T>>, item: T) {
    let mut guard = queue.lock().unwrap_or_else(PoisonError::into_inner);
    if guard.len() >= MAX_PENDING_PLOT_IMPORTS {
        guard.pop_front();
    }
    guard.push_back(item);
}

fn format_import_outcome_message(source: &str, outcome: &CsvImportOutcome) -> String {
    let count = outcome.datasets.len();
    let base = format!("Imported {count} dataset(s) from {source}");
    match (outcome.truncated_rows, outcome.truncated_series) {
        (true, true) => format!("{base} (capped at 10,000 rows and 64 series)"),
        (true, false) => format!("{base} (capped at 10,000 rows)"),
        (false, true) => format!("{base} (capped at 64 series)"),
        (false, false) => base,
    }
}

/// Drains any asynchronously imported datasets queued for `plot_id` (preserving sibling plots' entries).
#[must_use]
pub fn drain_imported_datasets(plot_id: egui::Id) -> Vec<ImportedDataset> {
    let mut datasets = Vec::new();
    let mut guard = pending_imports_queue()
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let mut remaining = VecDeque::with_capacity(guard.len());
    for (id, ds_list) in guard.drain(..) {
        if id == plot_id || id == egui::Id::NULL {
            datasets.extend(ds_list);
        } else {
            remaining.push_back((id, ds_list));
        }
    }
    *guard = remaining;
    datasets
}

/// Drains the latest asynchronous CSV import status `(message, is_error)` for `plot_id`.
#[must_use]
pub fn drain_import_status(plot_id: egui::Id) -> Option<(String, bool)> {
    let mut latest = None;
    let mut guard = pending_status_queue()
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let mut remaining = VecDeque::with_capacity(guard.len());
    for (id, msg, is_error) in guard.drain(..) {
        if id == plot_id || id == egui::Id::NULL {
            latest = Some((msg, is_error));
        } else {
            remaining.push_back((id, msg, is_error));
        }
    }
    *guard = remaining;
    latest
}

pub(crate) fn parse_immediate_csv_overlay(
    ctx: &egui::Context,
    source: &str,
    content: &str,
) -> Result<(Vec<ImportedDataset>, String), String> {
    match parse_csv_bounded(content) {
        Ok(outcome) => {
            let msg = format_import_outcome_message(source, &outcome);
            crate::emit_plot_live_message(ctx, msg.clone(), false);
            ctx.request_repaint();
            Ok((outcome.datasets, msg))
        }
        Err(CsvImportError::PayloadTooLarge {
            byte_len,
            max_bytes,
        }) => {
            let msg = format!(
                "Import failed for {source}: payload ({:.2} MiB) exceeds {:.1} MiB limit.",
                byte_len as f64 / (1024.0 * 1024.0),
                max_bytes as f64 / (1024.0 * 1024.0)
            );
            crate::emit_plot_live_message(ctx, msg.clone(), true);
            ctx.request_repaint();
            Err(msg)
        }
        Err(CsvImportError::NoValidData) => {
            let msg =
                format!("Invalid or empty CSV file: no numeric data points found in {source}");
            crate::emit_plot_live_message(ctx, msg.clone(), true);
            ctx.request_repaint();
            Err(msg)
        }
    }
}

pub(crate) fn enqueue_csv_text_import(
    ctx: &egui::Context,
    plot_id: egui::Id,
    source: &str,
    content: &str,
) -> (String, bool) {
    match parse_immediate_csv_overlay(ctx, source, content) {
        Ok((datasets, msg)) => {
            push_bounded(pending_imports_queue(), (plot_id, datasets));
            push_bounded(pending_status_queue(), (plot_id, msg.clone(), false));
            (msg, false)
        }
        Err(err) => {
            push_bounded(pending_status_queue(), (plot_id, err.clone(), true));
            (err, true)
        }
    }
}

pub(crate) fn process_plot_csv_import(
    ctx: &egui::Context,
    plot_id: egui::Id,
    outcome: Result<(String, String), String>,
) {
    match outcome {
        Ok((name, content)) => {
            let _ = enqueue_csv_text_import(ctx, plot_id, &name, &content);
        }
        Err(err) => {
            push_bounded(pending_status_queue(), (plot_id, err.clone(), true));
            crate::emit_plot_live_message(ctx, err, true);
            ctx.request_repaint();
        }
    }
}
