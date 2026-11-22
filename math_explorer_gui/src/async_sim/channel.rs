//! Non-blocking state snapshot and control event channel for simulation workers.
//!
//! Stores only the newest unread [`StateSnapshot`] in a single overwriteable slot while
//! preserving every [`WorkerStatusEvent`] and [`DiagnosticReport`] in an unbounded FIFO queue.

use super::{StateSnapshot, TimelineMetadata, WorkerStatusEvent};
use crate::framework::recovery::DiagnosticReport;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Ordered lifecycle or numerical divergence event emitted by a simulation worker.
#[derive(Clone, Debug)]
pub enum SimControlEvent {
    /// Worker lifecycle state transition notification.
    Status(WorkerStatusEvent),
    /// Numerical divergence diagnostic report captured before halting.
    Divergence(DiagnosticReport),
}

/// Error returned when sending to a channel whose [`SimStateReceiver`] has been dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimChannelClosed;

/// Batch of queued control events and the newest unread snapshot drained from [`SimStateReceiver`].
pub struct ChannelPollBatch {
    /// Ordered lifecycle and diagnostic events queued since the previous drain.
    pub events: Vec<SimControlEvent>,
    /// Most recent unread state snapshot, if one was published since the previous drain.
    pub latest_snapshot: Option<StateSnapshot>,
    /// Latest scrubbable timeline bounds reported by the worker, if time travel is enabled.
    pub timeline_metadata: Option<TimelineMetadata>,
    /// `true` when all [`SimStateSender`] handles have been dropped.
    pub disconnected: bool,
}

struct ChannelInner {
    latest_snapshot: Option<StateSnapshot>,
    timeline_metadata: Option<TimelineMetadata>,
    events: VecDeque<SimControlEvent>,
    sender_count: usize,
    receiver_alive: bool,
    seeded: bool,
}

/// Non-blocking sender half of a simulation state channel.
pub struct SimStateSender {
    inner: Arc<Mutex<ChannelInner>>,
}

impl Clone for SimStateSender {
    fn clone(&self) -> Self {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        guard.sender_count += 1;
        drop(guard);
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Drop for SimStateSender {
    fn drop(&mut self) {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        guard.sender_count = guard.sender_count.saturating_sub(1);
    }
}

impl SimStateSender {
    /// Marks the channel's initial seed as emitted and returns `true` if it was already seeded.
    pub fn mark_seeded(&self) -> bool {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let was_seeded = guard.seeded;
        guard.seeded = true;
        was_seeded
    }

    /// Stores `snapshot` as the newest unread snapshot, overwriting any unconsumed snapshot.
    pub fn send_snapshot(&self, snapshot: StateSnapshot) -> Result<(), SimChannelClosed> {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if !guard.receiver_alive {
            return Err(SimChannelClosed);
        }
        guard.latest_snapshot = Some(snapshot);
        Ok(())
    }

    /// Appends a [`WorkerStatusEvent`] to the unbounded control queue without blocking.
    pub fn send_status(&self, status: WorkerStatusEvent) -> Result<(), SimChannelClosed> {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if !guard.receiver_alive {
            return Err(SimChannelClosed);
        }
        guard.events.push_back(SimControlEvent::Status(status));
        Ok(())
    }

    /// Appends a [`DiagnosticReport`] to the unbounded control queue without blocking.
    pub fn send_divergence(&self, report: DiagnosticReport) -> Result<(), SimChannelClosed> {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if !guard.receiver_alive {
            return Err(SimChannelClosed);
        }
        guard.events.push_back(SimControlEvent::Divergence(report));
        Ok(())
    }

    /// Updates the latest [`TimelineMetadata`] exposed to the UI thread without blocking.
    pub fn set_timeline(&self, timeline: Option<TimelineMetadata>) {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if guard.receiver_alive {
            guard.timeline_metadata = timeline;
        }
    }

    /// Atomically publishes `snapshot` and appends `status` under a single lock acquisition.
    pub fn send_snapshot_and_status(
        &self,
        snapshot: StateSnapshot,
        status: WorkerStatusEvent,
    ) -> Result<(), SimChannelClosed> {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if !guard.receiver_alive {
            return Err(SimChannelClosed);
        }
        guard.latest_snapshot = Some(snapshot);
        guard.events.push_back(SimControlEvent::Status(status));
        Ok(())
    }
}

/// Non-blocking receiver half of a simulation state channel.
pub struct SimStateReceiver {
    inner: Arc<Mutex<ChannelInner>>,
}

impl Drop for SimStateReceiver {
    fn drop(&mut self) {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        guard.receiver_alive = false;
        guard.latest_snapshot = None;
        guard.events.clear();
    }
}

impl SimStateReceiver {
    /// Drains all queued control events and takes the newest unread snapshot without blocking.
    pub fn drain(&self) -> ChannelPollBatch {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let events = guard.events.drain(..).collect();
        let latest_snapshot = guard.latest_snapshot.take();
        let timeline_metadata = guard.timeline_metadata;
        let disconnected = guard.sender_count == 0;
        ChannelPollBatch {
            events,
            latest_snapshot,
            timeline_metadata,
            disconnected,
        }
    }

    /// Clears pending snapshots and control events for deterministic test state overrides.
    #[cfg(test)]
    pub fn clear(&self) {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        guard.latest_snapshot = None;
        guard.events.clear();
    }
}

/// Creates a paired non-blocking [`SimStateSender`] and [`SimStateReceiver`].
pub fn sim_state_channel() -> (SimStateSender, SimStateReceiver) {
    let inner = Arc::new(Mutex::new(ChannelInner {
        latest_snapshot: None,
        timeline_metadata: None,
        events: VecDeque::new(),
        sender_count: 1,
        receiver_alive: true,
        seeded: false,
    }));
    (
        SimStateSender {
            inner: Arc::clone(&inner),
        },
        SimStateReceiver { inner },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::RwLock;

    fn sample_snapshot(tag: f64) -> StateSnapshot {
        StateSnapshot {
            width: 1,
            height: 1,
            pixels: Arc::new(RwLock::new(vec![eframe::egui::Color32::WHITE])),
            custom_data: vec![tag],
            structured_data: None,
        }
    }

    #[test]
    fn overwrites_unread_snapshots_without_blocking_and_preserves_all_events() {
        let (tx, rx) = sim_state_channel();
        assert!(tx.send_status(WorkerStatusEvent::Running).is_ok());
        for step in 1..=64 {
            assert!(tx.send_snapshot(sample_snapshot(f64::from(step))).is_ok());
        }
        assert!(tx
            .send_status(WorkerStatusEvent::Faulted("NaN".to_string()))
            .is_ok());
        assert!(tx
            .send_divergence(DiagnosticReport::new(0.5, 64, "u", "NaN"))
            .is_ok());

        let batch = rx.drain();
        assert!(!batch.disconnected);
        assert_eq!(batch.latest_snapshot.unwrap().custom_data, vec![64.0]);
        assert_eq!(batch.events.len(), 3);
        assert!(matches!(
            batch.events[0],
            SimControlEvent::Status(WorkerStatusEvent::Running)
        ));
        assert!(matches!(
            batch.events[1],
            SimControlEvent::Status(WorkerStatusEvent::Faulted(_))
        ));
        assert!(matches!(batch.events[2], SimControlEvent::Divergence(_)));
    }

    #[test]
    fn reports_disconnection_and_receiver_closure() {
        let (tx, rx) = sim_state_channel();
        let tx_clone = tx.clone();
        drop(tx);
        assert!(!rx.drain().disconnected);
        drop(tx_clone);
        assert!(rx.drain().disconnected);

        let (tx2, rx2) = sim_state_channel();
        drop(rx2);
        assert_eq!(
            tx2.send_snapshot(sample_snapshot(1.0)),
            Err(SimChannelClosed)
        );
        assert_eq!(
            tx2.send_status(WorkerStatusEvent::Ready),
            Err(SimChannelClosed)
        );
    }
}
