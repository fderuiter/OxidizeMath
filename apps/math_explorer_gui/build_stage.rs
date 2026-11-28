//! Deterministic `OUT_DIR` staging helpers for `math_explorer_gui` build script.
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use epaint_default_fonts::{EMOJI_ICON, HACK_REGULAR, NOTO_EMOJI_REGULAR, UBUNTU_LIGHT};
use oxidize_core::path_utils;
use oxidize_core::workspace::find_workspace_root_from;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Non-ASCII UI symbols referenced internally by `egui`, `epaint`, and `eframe` widgets.
const EGUI_INTERNAL_SYMBOLS: &str = "°±²»ÄÅÖáäåéěπτ\u{2009}\u{200d}‐–•…‼⁇⁈⁉↺↻⇧−√∞≥⊗⌃⌘⌥⎗⎘⎙⏏⏩⏪⏭⏮⏴⏵⏶⏷⏸⏹⏺└■▶►◻☀☁★☆☐☑☜☝☞☟♂♡⚙⚠⛃⛶✒✔❓❤➡⟲⟳⬅⬆⬇⬈⬉⬊⬋⬌⬍⮨⮩⮪⮫、。々〉》」』】〕〗〙〜〟〻ぁぃぅぇぉしたっとのゃゅょゎゕゖ゠ァィゥェォッャュョヮヵヶ・ーヽヾㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ在文日本混章語\u{e618}\u{e61f}\u{e624}\u{e625}\u{f8ff}\u{fe0f}｝｠𝞃🌙🎑🎨🐛🐧👍💻📅📆📈📉📊📋📌📎📏📤📥📾🔀🔁🔃🔄🔆🔈🔉🔊🔍🔎🔗🔘🔠🔥🕓🖧🖩🖮🖱🖴🖵🖼🗀🗁🗋🗐🗑🗙😬🚫🤦🤷";

const DROPPED_SFNT_TABLES: &[[u8; 4]] = &[
    *b"DSIG", *b"GPOS", *b"GSUB", *b"LTSH", *b"TTFA", *b"VDMX", *b"cvt ", *b"fpgm", *b"gasp",
    *b"hdmx", *b"prep",
];

/// Computes `(ws_root_abs, rel_ws_root, gui_member_rel)` dynamically from `CARGO_MANIFEST_DIR`.
pub fn resolve_gui_workspace_paths() -> (PathBuf, PathBuf, String) {
    let manifest_dir = env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let ws_root_abs = find_workspace_root_from(&manifest_dir)
        .expect("Failed to locate Cargo workspace root from CARGO_MANIFEST_DIR");
    let canon_manifest = fs::canonicalize(&manifest_dir).unwrap_or_else(|_| manifest_dir.clone());
    let canon_ws = fs::canonicalize(&ws_root_abs).unwrap_or_else(|_| ws_root_abs.clone());
    let rel_from_ws = canon_manifest
        .strip_prefix(&canon_ws)
        .expect("CARGO_MANIFEST_DIR must reside inside the workspace root");
    let gui_member_rel = path_utils::normalize_path(rel_from_ws);
    let depth = rel_from_ws.components().count();
    let rel_ws_root = if depth == 0 {
        PathBuf::from(".")
    } else {
        let mut parent = PathBuf::new();
        for _ in 0..depth {
            parent.push("..");
        }
        parent
    };
    (ws_root_abs, rel_ws_root, gui_member_rel)
}

/// Writes `content` to `path` only when the file does not exist or its bytes differ,
/// preserving mtimes across warm builds and snapshot syncs.
pub fn write_if_changed(path: &Path, content: &[u8]) {
    if fs::read(path).ok().as_deref() == Some(content) {
        return;
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

fn stage_single_file(src_rel: &str, dst_path: &Path) {
    let src = Path::new(src_rel);
    if src.is_file() {
        println!("cargo:rerun-if-changed={src_rel}");
        if let Ok(bytes) = fs::read(src) {
            write_if_changed(dst_path, &bytes);
        }
    } else {
        write_if_changed(dst_path, b"");
    }
}

fn read_u16_be(buf: &[u8], off: usize) -> u16 {
    u16::from_be_bytes([buf[off], buf[off + 1]])
}

fn read_i16_be(buf: &[u8], off: usize) -> i16 {
    i16::from_be_bytes([buf[off], buf[off + 1]])
}

fn read_u32_be(buf: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn parse_sfnt_tables(raw: &[u8]) -> (u32, BTreeMap<[u8; 4], Vec<u8>>) {
    let sfnt_ver = read_u32_be(raw, 0);
    let num_tables = usize::from(read_u16_be(raw, 4));
    let mut tables = BTreeMap::new();
    for idx in 0..num_tables {
        let rec = 12 + idx * 16;
        let tag = [raw[rec], raw[rec + 1], raw[rec + 2], raw[rec + 3]];
        let off = read_u32_be(raw, rec + 8) as usize;
        let len = read_u32_be(raw, rec + 12) as usize;
        if !DROPPED_SFNT_TABLES.contains(&tag) {
            tables.insert(tag, raw[off..off + len].to_vec());
        }
    }
    (sfnt_ver, tables)
}

fn collect_cmap_fmt4(cmap: &[u8], sub_off: usize, keep: &BTreeSet<u32>, out: &mut BTreeSet<u16>) {
    let seg_count_x2 = usize::from(read_u16_be(cmap, sub_off + 6));
    let seg_count = seg_count_x2 / 2;
    let end_off = sub_off + 14;
    let start_off = end_off + seg_count_x2 + 2;
    let delta_off = start_off + seg_count_x2;
    let range_off = delta_off + seg_count_x2;
    for s in 0..seg_count {
        let end_c = u32::from(read_u16_be(cmap, end_off + s * 2));
        let start_c = u32::from(read_u16_be(cmap, start_off + s * 2));
        let id_delta = i32::from(read_i16_be(cmap, delta_off + s * 2));
        let id_range = usize::from(read_u16_be(cmap, range_off + s * 2));
        for &cp in keep.range(start_c..=end_c) {
            let gid = if id_range == 0 {
                ((cp as i32 + id_delta) & 0xFFFF) as u16
            } else {
                let roff = range_off + s * 2 + id_range + ((cp - start_c) as usize) * 2;
                let g = if roff + 2 <= cmap.len() {
                    read_u16_be(cmap, roff)
                } else {
                    0
                };
                if g != 0 {
                    ((i32::from(g) + id_delta) & 0xFFFF) as u16
                } else {
                    0
                }
            };
            out.insert(gid);
        }
    }
}

fn collect_kept_glyph_ids(cmap: &[u8], keep_cps: &BTreeSet<u32>) -> BTreeSet<u16> {
    let mut kept = BTreeSet::from([0u16]);
    let num_tables = usize::from(read_u16_be(cmap, 2));
    for idx in 0..num_tables {
        let sub_off = read_u32_be(cmap, 8 + idx * 8) as usize;
        if sub_off + 4 > cmap.len() {
            continue;
        }
        match read_u16_be(cmap, sub_off) {
            4 => collect_cmap_fmt4(cmap, sub_off, keep_cps, &mut kept),
            12 => {
                let groups = read_u32_be(cmap, sub_off + 12) as usize;
                for g in 0..groups {
                    let goff = sub_off + 16 + g * 12;
                    let start_c = read_u32_be(cmap, goff);
                    let end_c = read_u32_be(cmap, goff + 4);
                    let start_gid = read_u32_be(cmap, goff + 8);
                    for &cp in keep_cps.range(start_c..=end_c) {
                        if let Ok(gid) = u16::try_from(start_gid + (cp - start_c)) {
                            kept.insert(gid);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    kept
}

fn expand_composite_glyphs(
    glyf: &[u8],
    offsets: &[usize],
    num_glyphs: usize,
    kept: &mut BTreeSet<u16>,
) {
    let mut stack: Vec<u16> = kept.iter().copied().collect();
    while let Some(gid) = stack.pop() {
        let idx = usize::from(gid);
        if idx >= num_glyphs || offsets[idx + 1] - offsets[idx] < 10 {
            continue;
        }
        let (s, e) = (offsets[idx], offsets[idx + 1]);
        if read_i16_be(glyf, s) >= 0 {
            continue;
        }
        let mut pos = s + 10;
        while pos + 4 <= e {
            let flags = read_u16_be(glyf, pos);
            let comp_gid = read_u16_be(glyf, pos + 2);
            pos += 4 + if flags & 0x0001 != 0 { 4 } else { 2 };
            pos += if flags & 0x0008 != 0 {
                2
            } else if flags & 0x0040 != 0 {
                4
            } else if flags & 0x0080 != 0 {
                8
            } else {
                0
            };
            if kept.insert(comp_gid) {
                stack.push(comp_gid);
            }
            if flags & 0x0020 == 0 {
                break;
            }
        }
    }
}

fn rebuild_glyf_and_loca(
    glyf: &[u8],
    offsets: &[usize],
    num_glyphs: usize,
    kept: &BTreeSet<u16>,
) -> (Vec<u8>, Vec<u8>, i16) {
    let mut new_glyf = Vec::new();
    let mut new_offsets = Vec::with_capacity(num_glyphs + 1);
    new_offsets.push(0usize);
    for gid in 0..num_glyphs {
        if kept.contains(&(gid as u16)) {
            let chunk = &glyf[offsets[gid]..offsets[gid + 1]];
            if chunk.len() >= 10 {
                let nc = read_i16_be(chunk, 0);
                let ins_off = 10 + (nc.max(0) as usize) * 2;
                if nc > 0 && ins_off + 2 <= chunk.len() {
                    let ins_len = usize::from(read_u16_be(chunk, ins_off));
                    if ins_len > 0 && ins_off + 2 + ins_len <= chunk.len() {
                        new_glyf.extend_from_slice(&chunk[..ins_off]);
                        new_glyf.extend_from_slice(&[0, 0]);
                        new_glyf.extend_from_slice(&chunk[ins_off + 2 + ins_len..]);
                    } else {
                        new_glyf.extend_from_slice(chunk);
                    }
                } else {
                    new_glyf.extend_from_slice(chunk);
                }
            } else {
                new_glyf.extend_from_slice(chunk);
            }
            while new_glyf.len() % 4 != 0 {
                new_glyf.push(0);
            }
        }
        new_offsets.push(new_glyf.len());
    }
    let mut new_loca = Vec::new();
    let loc_fmt = if new_glyf.len() / 2 <= usize::from(u16::MAX) {
        for &off in &new_offsets {
            new_loca.extend_from_slice(&((off / 2) as u16).to_be_bytes());
        }
        0i16
    } else {
        for &off in &new_offsets {
            new_loca.extend_from_slice(&(off as u32).to_be_bytes());
        }
        1i16
    };
    (new_glyf, new_loca, loc_fmt)
}

fn sfnt_checksum(data: &[u8]) -> u32 {
    let mut sum = 0u32;
    let (chunks, rem) = data.as_chunks::<4>();
    for &chunk in chunks {
        sum = sum.wrapping_add(u32::from_be_bytes(chunk));
    }
    if !rem.is_empty() {
        let mut tail = [0u8; 4];
        tail[..rem.len()].copy_from_slice(rem);
        sum = sum.wrapping_add(u32::from_be_bytes(tail));
    }
    sum
}

fn encode_sfnt(sfnt_ver: u32, tables: &BTreeMap<[u8; 4], Vec<u8>>) -> Vec<u8> {
    let n = tables.len() as u16;
    let max_pow2 = if n == 0 { 1 } else { 1u16 << n.ilog2() };
    let entry_sel = if n == 0 { 0 } else { n.ilog2() as u16 };
    let search_range = max_pow2 * 16;
    let range_shift = n * 16 - search_range;

    let mut out = Vec::new();
    out.extend_from_slice(&sfnt_ver.to_be_bytes());
    out.extend_from_slice(&n.to_be_bytes());
    out.extend_from_slice(&search_range.to_be_bytes());
    out.extend_from_slice(&entry_sel.to_be_bytes());
    out.extend_from_slice(&range_shift.to_be_bytes());
    let dir_pos = out.len();
    out.resize(dir_pos + usize::from(n) * 16, 0);

    let mut head_off = 0usize;
    for (idx, (tag, data)) in tables.iter().enumerate() {
        let off = out.len();
        if tag == b"head" {
            head_off = off;
        }
        let rec = dir_pos + idx * 16;
        out[rec..rec + 4].copy_from_slice(tag);
        out[rec + 4..rec + 8].copy_from_slice(&sfnt_checksum(data).to_be_bytes());
        out[rec + 8..rec + 12].copy_from_slice(&(off as u32).to_be_bytes());
        out[rec + 12..rec + 16].copy_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(data);
        while out.len() % 4 != 0 {
            out.push(0);
        }
    }
    let adj = 0xB1B0_AFBAu32.wrapping_sub(sfnt_checksum(&out));
    out[head_off + 8..head_off + 12].copy_from_slice(&adj.to_be_bytes());
    out
}

fn compact_ttf(raw: &[u8], keep_cps: &BTreeSet<u32>) -> Vec<u8> {
    let (sfnt_ver, mut tables) = parse_sfnt_tables(raw);
    let loc_fmt = read_i16_be(&tables[b"head"], 50);
    let num_glyphs = usize::from(read_u16_be(&tables[b"maxp"], 4));
    let loca = &tables[b"loca"];
    let offsets: Vec<usize> = (0..=num_glyphs)
        .map(|i| {
            if loc_fmt == 0 {
                usize::from(read_u16_be(loca, i * 2)) * 2
            } else {
                read_u32_be(loca, i * 4) as usize
            }
        })
        .collect();

    let mut kept = collect_kept_glyph_ids(&tables[b"cmap"], keep_cps);
    expand_composite_glyphs(&tables[b"glyf"], &offsets, num_glyphs, &mut kept);
    let (new_glyf, new_loca, new_loc_fmt) =
        rebuild_glyf_and_loca(&tables[b"glyf"], &offsets, num_glyphs, &kept);
    tables.insert(*b"glyf", new_glyf);
    tables.insert(*b"loca", new_loca);

    if let Some(head) = tables.get_mut(b"head") {
        head[8..12].copy_from_slice(&0u32.to_be_bytes());
        head[50..52].copy_from_slice(&new_loc_fmt.to_be_bytes());
    }
    if let Some(post) = tables.get_mut(b"post") {
        if post.len() > 32 {
            post.truncate(32);
            post[0..4].copy_from_slice(&0x0003_0000u32.to_be_bytes());
        }
    }
    encode_sfnt(sfnt_ver, &tables)
}

fn scan_dir_non_ascii(dir: &Path, cps: &mut BTreeSet<u32>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        if path.is_dir() {
            if !matches!(name_str, "target" | "dist" | ".git" | "node_modules") {
                scan_dir_non_ascii(&path, cps);
            }
        } else if matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("rs" | "md" | "toml" | "html" | "js" | "json" | "bib" | "tex")
        ) {
            if let Ok(text) = fs::read_to_string(&path) {
                cps.extend(text.chars().map(u32::from).filter(|&cp| cp >= 128));
            }
        }
    }
}

fn stage_wasm_fonts(ws_root: &Path, out_dir: &Path) {
    let default_assets = [
        ("Hack-Regular.ttf", HACK_REGULAR, false),
        ("NotoEmoji-Regular.ttf", NOTO_EMOJI_REGULAR, true),
        ("Ubuntu-Light.ttf", UBUNTU_LIGHT, false),
        ("emoji-icon-font.ttf", EMOJI_ICON, true),
    ];
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("wasm32") {
        assert!(default_assets.iter().all(|(_, data, _)| !data.is_empty()));
        return;
    }

    let mut non_ascii: BTreeSet<u32> = EGUI_INTERNAL_SYMBOLS.chars().map(u32::from).collect();
    for sub in ["crates", "apps/math_explorer_gui", "egui_plot", "papers"] {
        scan_dir_non_ascii(&ws_root.join(sub), &mut non_ascii);
    }
    if let Ok(readme) = fs::read_to_string(ws_root.join("README.md")) {
        non_ascii.extend(readme.chars().map(u32::from).filter(|&cp| cp >= 128));
    }

    let mut text_cps: BTreeSet<u32> = (0x0000..=0x03FF)
        .chain(0x2000..=0x22FF)
        .chain(0x25A0..=0x25FF)
        .chain([0xFFFD])
        .collect();
    text_cps.extend(&non_ascii);

    let mut emoji_cps: BTreeSet<u32> = (0x0020..=0x007E).chain([0x25FB, 0xFFFD]).collect();
    emoji_cps.extend(&non_ascii);

    let font_dir = out_dir.join("wasm_fonts");
    for (name, raw, is_emoji) in default_assets {
        let compacted = compact_ttf(raw, if is_emoji { &emoji_cps } else { &text_cps });
        write_if_changed(&font_dir.join(name), &compacted);
    }
}

/// Stages compile-time embedded assets into `out_dir` so `rustc` never depends on
/// `env!("CARGO_MANIFEST_DIR")`.
pub fn stage_gui_inputs_to_out_dir(out_dir: &Path) {
    println!("cargo:rerun-if-changed=build_stage.rs");
    let (ws_root_abs, rel_ws_root, _) = resolve_gui_workspace_paths();
    let cert_rel = path_utils::normalize_path(rel_ws_root.join("Verification_Certificate.json"));
    let bib_rel = path_utils::normalize_path(rel_ws_root.join("papers/references.bib"));

    stage_single_file("sw.js", &out_dir.join("sw.js"));
    stage_single_file(&cert_rel, &out_dir.join("Verification_Certificate.json"));
    stage_single_file(&bib_rel, &out_dir.join("references.bib"));
    stage_wasm_fonts(&ws_root_abs, out_dir);
}
