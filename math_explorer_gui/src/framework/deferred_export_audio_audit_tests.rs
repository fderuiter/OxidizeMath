#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::deferred_export_audit_tests::run_deferred_export_frame;
use super::*;

fn verify_scala_exports(fw: &mut SimulationFramework, meta_id: &str, caps: ExportCapabilities) {
    if caps.scala {
        assert!(
            caps.scala_disabled_reason.is_empty(),
            "{meta_id} scala reason must be empty"
        );
        run_deferred_export_frame(fw, ExportFormat::ScalaScl, false);
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        let scl = fw
            .export_queue
            .last_exported_content
            .as_deref()
            .expect("SCL text");
        assert!(scl.contains('!') && scl.len() > 20);

        run_deferred_export_frame(fw, ExportFormat::ScalaKbm, false);
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        let kbm = fw
            .export_queue
            .last_exported_content
            .as_deref()
            .expect("KBM text");
        assert!(kbm.contains("Mapping") && kbm.contains("127"));
    } else {
        assert!(!caps.scala_disabled_reason.is_empty());
    }
}

fn verify_audio_tool_exports(
    meta_id: &str,
    tool: Box<dyn InteractiveTool>,
    caps: ExportCapabilities,
) {
    let mut fw = SimulationFramework::new("__audit_audio__");
    fw.active_tool = Some(tool);

    if caps.wav {
        assert!(
            caps.wav_disabled_reason.is_empty(),
            "{meta_id} wav reason must be empty"
        );
        run_deferred_export_frame(&mut fw, ExportFormat::Wav, false);
        assert_eq!(
            fw.export_status_severity,
            ExportStatusSeverity::Success,
            "WAV export failed for {meta_id}: {:?}",
            fw.export_status_message
        );
        let wav = fw
            .export_queue
            .last_exported_bytes
            .as_deref()
            .expect("WAV bytes");
        assert!(wav.starts_with(b"RIFF") && &wav[8..12] == b"WAVE");
    } else {
        assert!(!caps.wav_disabled_reason.is_empty());
    }

    if caps.midi {
        assert!(
            caps.midi_disabled_reason.is_empty(),
            "{meta_id} midi reason must be empty"
        );
        run_deferred_export_frame(&mut fw, ExportFormat::Midi, false);
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        let mid = fw
            .export_queue
            .last_exported_bytes
            .as_deref()
            .expect("MIDI bytes");
        assert!(mid.starts_with(b"MThd"));
    } else {
        assert!(!caps.midi_disabled_reason.is_empty());
    }

    verify_scala_exports(&mut fw, meta_id, caps);
}

#[test]
fn test_all_audio_midi_and_scala_tools_deferred_exports_and_capabilities() {
    crate::audio::set_offline_duration_secs(1.0);
    crate::audio::set_offline_sample_rate(44_100);

    let mut wav_count = 0_usize;
    let mut midi_count = 0_usize;
    let mut scala_ids = Vec::new();

    for meta in inventory::iter::<crate::framework::ToolMetadata> {
        let tool = (meta.build)();
        let caps = tool.export_capabilities();
        if caps.wav {
            wav_count += 1;
        }
        if caps.midi {
            midi_count += 1;
        }
        if caps.scala {
            scala_ids.push(meta.id);
        }
        verify_audio_tool_exports(meta.id, tool, caps);
    }

    assert!(wav_count >= 9, "Expected >= 9 WAV tools, got {wav_count}");
    assert!(
        midi_count >= 9,
        "Expected >= 9 MIDI tools, got {midi_count}"
    );
    for required in [
        "ChladniWaveTankTool",
        "FractranMachineTool",
        "FactorizationTool",
    ] {
        assert!(
            scala_ids.contains(&required),
            "Missing microtonal studio exports on {required} (found {scala_ids:?})"
        );
    }
}
