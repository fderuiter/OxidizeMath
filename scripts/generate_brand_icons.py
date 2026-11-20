#!/usr/bin/env python3
"""Deterministically rasterize Math Explorer .brand-mark icons and social assets.

Generates PNG, ICO (PNG-compressed frames), and ICNS assets in
math_explorer_gui/icons/ from the inline .brand-mark wave geometry in
math_explorer_gui/index.html without timestamps or platform-dependent metadata.
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

# Dark .brand-mark palette from math_explorer_gui/index.html:L83-L99
SHELL_BG = (18, 18, 20, 255)        # #121214
BADGE_FILL = (23, 37, 84, 255)      # #172554
BADGE_STROKE = (59, 130, 246, 255)  # #3b82f6
AXES_RGB = (96, 165, 250)           # #60a5fa (stroke-opacity: 0.5)
WAVE_STROKE = (255, 255, 255, 255)  # #ffffff
NODE_FILL = (56, 189, 248, 255)     # #38bdf8
GRID_RGB = (59, 130, 246)           # subtle OG coordinate grid


def clamp01(val: float) -> float:
    if val <= 0.0:
        return 0.0
    if val >= 1.0:
        return 1.0
    return val


def blend_over(
    dst: tuple[int, int, int, int],
    src_rgb: tuple[int, int, int],
    src_alpha: float,
) -> tuple[int, int, int, int]:
    """Porter-Duff source-over compositing in straight RGBA."""
    sa = clamp01(src_alpha)
    if sa <= 0.0:
        return dst
    if sa >= 1.0:
        return (src_rgb[0], src_rgb[1], src_rgb[2], 255)
    da = dst[3] / 255.0
    out_a = sa + da * (1.0 - sa)
    if out_a <= 1e-9:
        return (0, 0, 0, 0)
    inv = 1.0 - sa
    r = int(round((src_rgb[0] * sa + dst[0] * da * inv) / out_a))
    g = int(round((src_rgb[1] * sa + dst[1] * da * inv) / out_a))
    b = int(round((src_rgb[2] * sa + dst[2] * da * inv) / out_a))
    a = int(round(out_a * 255.0))
    return (
        max(0, min(255, r)),
        max(0, min(255, g)),
        max(0, min(255, b)),
        max(0, min(255, a)),
    )


def sample_cubic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    steps: int,
) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for i in range(steps + 1):
        t = i / steps
        mt = 1.0 - t
        w0 = mt * mt * mt
        w1 = 3.0 * mt * mt * t
        w2 = 3.0 * mt * t * t
        w3 = t * t * t
        pts.append((
            w0 * p0[0] + w1 * p1[0] + w2 * p2[0] + w3 * p3[0],
            w0 * p0[1] + w1 * p1[1] + w2 * p2[1] + w3 * p3[1],
        ))
    return pts


def build_wave_segments(
    cx: float,
    cy: float,
    scale: float,
    steps_per_curve: int = 48,
) -> list[tuple[float, float, float, float, float, float, float]]:
    """Precompute line segments (ax, ay, dx, dy, len2, x_lo, x_hi) in pixel space.

    Matches <path d="M 14 36 C 20 18, 26 18, 32 32 C 38 46, 44 46, 50 28"/>
    centered around (32, 32) in the 64x64 viewBox.
    """
    def tr(pt: tuple[float, float]) -> tuple[float, float]:
        return (cx + (pt[0] - 32.0) * scale, cy + (pt[1] - 32.0) * scale)

    c1 = sample_cubic_bezier(tr((14.0, 36.0)), tr((20.0, 18.0)), tr((26.0, 18.0)), tr((32.0, 32.0)), steps_per_curve)
    c2 = sample_cubic_bezier(tr((32.0, 32.0)), tr((38.0, 46.0)), tr((44.0, 46.0)), tr((50.0, 28.0)), steps_per_curve)
    pts = c1 + c2[1:]
    segs: list[tuple[float, float, float, float, float, float, float]] = []
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        dx = bx - ax
        dy = by - ay
        len2 = dx * dx + dy * dy
        segs.append((ax, ay, dx, dy, len2, min(ax, bx), max(ax, bx)))
    return segs


def min_dist_to_wave(
    x: float,
    y: float,
    limit: float,
    segs: list[tuple[float, float, float, float, float, float, float]],
) -> float:
    best_d2 = (limit + 1.0) * (limit + 1.0)
    for ax, ay, dx, dy, len2, x_lo, x_hi in segs:
        if x < x_lo - limit or x > x_hi + limit:
            continue
        wx = x - ax
        wy = y - ay
        if len2 > 1e-12:
            t = (wx * dx + wy * dy) / len2
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            px = wx - t * dx
            py = wy - t * dy
        else:
            px = wx
            py = wy
        d2 = px * px + py * py
        if d2 < best_d2:
            best_d2 = d2
    return math.sqrt(best_d2)


def rounded_rect_sdf(
    x: float,
    y: float,
    cx: float,
    cy: float,
    hx: float,
    hy: float,
    radius: float,
) -> float:
    qx = abs(x - cx) - (hx - radius)
    qy = abs(y - cy) - (hy - radius)
    out = math.hypot(max(qx, 0.0), max(qy, 0.0))
    inside = min(max(qx, qy), 0.0)
    return out + inside - radius


def render_emblem_overlay(
    buf: bytearray,
    width: int,
    height: int,
    cx: float,
    cy: float,
    emblem_scale: float,
) -> None:
    """Composite .brand-axes, .brand-wave, and .brand-node centered at (cx, cy)."""
    hw_axis = max(1.0 * emblem_scale, 0.65)
    hw_wave = max(2.25 * emblem_scale, 0.85)
    r_node = max(4.0 * emblem_scale, 1.35)

    span = 18.0 * emblem_scale
    ax_x1, ax_x2 = cx - span, cx + span
    ax_y1, ax_y2 = cy - span, cy + span

    segs = build_wave_segments(cx, cy, emblem_scale)
    bound_pad = max(hw_wave, hw_axis, r_node) + 2.0
    x_min = max(0, int(math.floor(cx - span - bound_pad)))
    x_max = min(width, int(math.ceil(cx + span + bound_pad)) + 1)
    y_min = max(0, int(math.floor(cy - span - bound_pad)))
    y_max = min(height, int(math.ceil(cy + span + bound_pad)) + 1)

    wave_y_lo = cy - 11.5 * emblem_scale - hw_wave - 1.5
    wave_y_hi = cy + 11.5 * emblem_scale + hw_wave + 1.5

    for py in range(y_min, y_max):
        y = py + 0.5
        row_off = py * width * 4
        in_wave_y = wave_y_lo <= y <= wave_y_hi
        dy_h = abs(y - cy)
        dy_v = max(ax_y1 - y, 0.0, y - ax_y2)
        for px in range(x_min, x_max):
            x = px + 0.5
            idx = row_off + px * 4
            px_rgba = (buf[idx], buf[idx + 1], buf[idx + 2], buf[idx + 3])

            # Horizontal axis: (14, 32) -> (50, 32)
            if dy_h <= hw_axis + 0.6:
                dx_h = max(ax_x1 - x, 0.0, x - ax_x2)
                d_h = math.hypot(dx_h, dy_h) - hw_axis
                if d_h < 0.5:
                    px_rgba = blend_over(px_rgba, AXES_RGB, 0.5 * (0.5 - d_h))

            # Vertical axis: (32, 14) -> (32, 50)
            dx_v = abs(x - cx)
            if dx_v <= hw_axis + 0.6:
                d_v = math.hypot(dx_v, dy_v) - hw_axis
                if d_v < 0.5:
                    px_rgba = blend_over(px_rgba, AXES_RGB, 0.5 * (0.5 - d_v))

            # Cubic Bezier wave
            if in_wave_y:
                d_wave = min_dist_to_wave(x, y, hw_wave + 0.6, segs) - hw_wave
                if d_wave < 0.5:
                    px_rgba = blend_over(px_rgba, WAVE_STROKE[:3], 0.5 - d_wave)

            # Center node circle: (32, 32) r=4
            d_node = math.hypot(x - cx, y - cy) - r_node
            if d_node < 0.5:
                px_rgba = blend_over(px_rgba, NODE_FILL[:3], 0.5 - d_node)

            buf[idx] = px_rgba[0]
            buf[idx + 1] = px_rgba[1]
            buf[idx + 2] = px_rgba[2]
            buf[idx + 3] = px_rgba[3]


def render_standard_icon(size: int) -> bytes:
    """Render transparent-corner .brand-mark icon at size x size."""
    buf = bytearray(size * size * 4)
    scale = size / 64.0
    cx = cy = size * 0.5
    half_box = 28.0 * scale
    radius = 14.0 * scale
    hw_stroke = max(1.5 * scale, 1.05 if size <= 32 else 1.5 * scale)

    for py in range(size):
        y = py + 0.5
        row_off = py * size * 4
        for px in range(size):
            x = px + 0.5
            d_rect = rounded_rect_sdf(x, y, cx, cy, half_box, half_box, radius)
            if d_rect >= hw_stroke + 0.5:
                continue
            px_rgba = (0, 0, 0, 0)
            if d_rect < 0.5:
                px_rgba = blend_over(px_rgba, BADGE_FILL[:3], 0.5 - d_rect)
            d_stroke = abs(d_rect) - hw_stroke
            if d_stroke < 0.5:
                px_rgba = blend_over(px_rgba, BADGE_STROKE[:3], 0.5 - d_stroke)
            idx = row_off + px * 4
            buf[idx] = px_rgba[0]
            buf[idx + 1] = px_rgba[1]
            buf[idx + 2] = px_rgba[2]
            buf[idx + 3] = px_rgba[3]

    render_emblem_overlay(buf, size, size, cx, cy, scale)
    return encode_png_rgba(size, size, bytes(buf))


def render_apple_touch_icon(size: int = 180) -> bytes:
    """Render opaque 180x180 apple-touch-icon with full-bleed #172554 background."""
    buf = bytearray(BADGE_FILL * (size * size))
    cx = cy = size * 0.5
    scale = size / 64.0
    # Inset rounded #3b82f6 border frame so iOS squircle mask preserves full-bleed navy
    half_box = 26.5 * scale
    radius = 13.0 * scale
    hw_stroke = 1.5 * scale

    for py in range(size):
        y = py + 0.5
        row_off = py * size * 4
        for px in range(size):
            x = px + 0.5
            d_rect = rounded_rect_sdf(x, y, cx, cy, half_box, half_box, radius)
            d_stroke = abs(d_rect) - hw_stroke
            if d_stroke < 0.5:
                idx = row_off + px * 4
                px_rgba = blend_over(BADGE_FILL, BADGE_STROKE[:3], 0.5 - d_stroke)
                buf[idx] = px_rgba[0]
                buf[idx + 1] = px_rgba[1]
                buf[idx + 2] = px_rgba[2]
                buf[idx + 3] = 255

    render_emblem_overlay(buf, size, size, cx, cy, scale * 0.94)
    return encode_png_rgba(size, size, bytes(buf))


def render_maskable_icon(size: int = 512) -> bytes:
    """Render opaque 512x512 maskable icon keeping badge/wave/axes inside inner 80% safe circle."""
    buf = bytearray(BADGE_FILL * (size * size))
    cx = cy = size * 0.5
    # Inner 80% safe circle has radius 0.40 * 512 = 204.8 px.
    # At scale = 0.70 * (size / 64), the rounded-rect corner outer edge sits at
    # (hypot(14, 14) + 14 + 1.5) * 5.6 = 197.7 px <= 204.8 px, and wave/axes <= 116 px.
    emblem_scale = (size / 64.0) * 0.70
    half_box = 28.0 * emblem_scale
    radius = 14.0 * emblem_scale
    hw_stroke = 1.5 * emblem_scale
    bound = int(math.ceil(half_box + hw_stroke + 2.0))
    lo = max(0, int(cx) - bound)
    hi = min(size, int(cx) + bound + 1)

    for py in range(lo, hi):
        y = py + 0.5
        row_off = py * size * 4
        for px in range(lo, hi):
            x = px + 0.5
            d_rect = rounded_rect_sdf(x, y, cx, cy, half_box, half_box, radius)
            d_stroke = abs(d_rect) - hw_stroke
            if d_stroke < 0.5:
                idx = row_off + px * 4
                px_rgba = blend_over(BADGE_FILL, BADGE_STROKE[:3], 0.5 - d_stroke)
                buf[idx] = px_rgba[0]
                buf[idx + 1] = px_rgba[1]
                buf[idx + 2] = px_rgba[2]
                buf[idx + 3] = 255

    render_emblem_overlay(buf, size, size, cx, cy, emblem_scale)
    return encode_png_rgba(size, size, bytes(buf))


def render_og_image(width: int = 1200, height: int = 630) -> bytes:
    """Render deterministic 1200x630 Open Graph preview banner (<20 KB PNG)."""
    buf = bytearray(SHELL_BG * (width * height))
    cx = width * 0.5
    cy = height * 0.5

    # Subtle coordinate grid lines across the dark #121214 backdrop
    grid_step = 90
    grid_pixel = blend_over(SHELL_BG, GRID_RGB, 0.14)
    axis_pixel = blend_over(SHELL_BG, AXES_RGB, 0.24)

    for py in range(height):
        row_off = py * width * 4
        is_mid_h = abs(py - int(cy)) <= 1
        is_grid_h = (py - 45) % grid_step == 0
        if is_mid_h or is_grid_h:
            p = axis_pixel if is_mid_h else grid_pixel
            buf[row_off : row_off + width * 4] = bytes(p) * width

    for px in range(60, width, grid_step):
        is_mid_v = abs(px - int(cx)) <= 1
        p = bytes(axis_pixel if is_mid_v else grid_pixel)
        for py in range(height):
            idx = (py * width + px) * 4
            buf[idx : idx + 4] = p

    # Prominent centered .brand-mark badge (384x384 -> scale = 6.0)
    badge_size = 384.0
    scale = badge_size / 64.0
    half_box = 28.0 * scale
    radius = 14.0 * scale
    hw_stroke = 1.5 * scale

    bx_min = max(0, int(math.floor(cx - half_box - hw_stroke - 2.0)))
    bx_max = min(width, int(math.ceil(cx + half_box + hw_stroke + 2.0)) + 1)
    by_min = max(0, int(math.floor(cy - half_box - hw_stroke - 2.0)))
    by_max = min(height, int(math.ceil(cy + half_box + hw_stroke + 2.0)) + 1)

    for py in range(by_min, by_max):
        y = py + 0.5
        row_off = py * width * 4
        for px in range(bx_min, bx_max):
            x = px + 0.5
            d_rect = rounded_rect_sdf(x, y, cx, cy, half_box, half_box, radius)
            if d_rect >= hw_stroke + 0.5:
                continue
            idx = row_off + px * 4
            px_rgba = (buf[idx], buf[idx + 1], buf[idx + 2], 255)
            if d_rect < 0.5:
                px_rgba = blend_over(px_rgba, BADGE_FILL[:3], 0.5 - d_rect)
            d_stroke = abs(d_rect) - hw_stroke
            if d_stroke < 0.5:
                px_rgba = blend_over(px_rgba, BADGE_STROKE[:3], 0.5 - d_stroke)
            buf[idx] = px_rgba[0]
            buf[idx + 1] = px_rgba[1]
            buf[idx + 2] = px_rgba[2]
            buf[idx + 3] = 255

    render_emblem_overlay(buf, width, height, cx, cy, scale)
    return encode_png_rgba(width, height, bytes(buf))


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _filter_sub_row(row: bytes, bpp: int = 4) -> bytes:
    out = bytearray(len(row) + 1)
    out[0] = 1  # Sub filter
    out[1 : 1 + bpp] = row[:bpp]
    for i in range(bpp, len(row)):
        out[1 + i] = (row[i] - row[i - bpp]) & 0xFF
    return bytes(out)


def _filter_up_row(row: bytes, prev: bytes) -> bytes:
    out = bytearray(len(row) + 1)
    out[0] = 2  # Up filter
    for i in range(len(row)):
        out[1 + i] = (row[i] - prev[i]) & 0xFF
    return bytes(out)


def _filter_paeth_row(row: bytes, prev: bytes, bpp: int = 4) -> bytes:
    out = bytearray(len(row) + 1)
    out[0] = 4  # Paeth filter
    for i in range(bpp):
        out[1 + i] = (row[i] - prev[i]) & 0xFF
    for i in range(bpp, len(row)):
        a = row[i - bpp]
        b = prev[i]
        c = prev[i - bpp]
        p = a + b - c
        pa = abs(p - a)
        pb = abs(p - b)
        pc = abs(p - c)
        pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
        out[1 + i] = (row[i] - pr) & 0xFF
    return bytes(out)


def encode_png_rgba(width: int, height: int, rgba: bytes) -> bytes:
    """Deterministically encode RGBA8888 pixels to PNG without timestamp chunks."""
    stride = width * 4
    if len(rgba) != stride * height:
        raise ValueError(f"Expected {stride * height} RGBA bytes, got {len(rgba)}")

    raw_rows = bytearray()
    prev_row = bytes(stride)
    for y in range(height):
        row = rgba[y * stride : (y + 1) * stride]
        if row == prev_row:
            raw_rows.append(2)
            raw_rows.extend(bytes(stride))
        else:
            cand_none = b"\x00" + row
            cand_sub = _filter_sub_row(row, 4)
            cand_up = _filter_up_row(row, prev_row)
            cand_paeth = _filter_paeth_row(row, prev_row, 4)
            best = min(
                (cand_paeth, cand_up, cand_sub, cand_none),
                key=lambda c: sum(b if b < 128 else 256 - b for b in c[1:]),
            )
            raw_rows.extend(best)
        prev_row = row

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    idat = zlib.compress(bytes(raw_rows), level=9)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def encode_ico(frames: list[tuple[int, bytes]]) -> bytes:
    """Pack PNG-encoded icon frames into a Windows .ico container."""
    header = struct.pack("<HHH", 0, 1, len(frames))
    entries = bytearray()
    payload = bytearray()
    offset = 6 + 16 * len(frames)

    for dim, png_bytes in frames:
        w_byte = 0 if dim == 256 else dim
        h_byte = 0 if dim == 256 else dim
        entries.extend(
            struct.pack(
                "<BBBBHHII",
                w_byte,
                h_byte,
                0,
                0,
                1,
                32,
                len(png_bytes),
                offset,
            )
        )
        payload.extend(png_bytes)
        offset += len(png_bytes)

    return header + bytes(entries) + bytes(payload)


def encode_icns(chunks: list[tuple[bytes, bytes]]) -> bytes:
    """Pack PNG-encoded icon chunks into a macOS .icns container."""
    body = bytearray()
    for ostype, png_bytes in chunks:
        if len(ostype) != 4:
            raise ValueError(f"ICNS OSType must be 4 bytes, got {ostype!r}")
        body.extend(ostype)
        body.extend(struct.pack(">I", 8 + len(png_bytes)))
        body.extend(png_bytes)
    return b"icns" + struct.pack(">I", 8 + len(body)) + bytes(body)


def generate_all(icons_dir: Path) -> None:
    icons_dir.mkdir(parents=True, exist_ok=True)

    png_16 = render_standard_icon(16)
    png_32 = render_standard_icon(32)
    png_48 = render_standard_icon(48)
    png_128 = render_standard_icon(128)
    png_192 = render_standard_icon(192)
    png_256 = render_standard_icon(256)
    png_512 = render_standard_icon(512)

    png_180 = render_apple_touch_icon(180)
    png_512_maskable = render_maskable_icon(512)
    png_og = render_og_image(1200, 630)

    ico_bytes = encode_ico([
        (16, png_16),
        (32, png_32),
        (48, png_48),
        (256, png_256),
    ])
    icns_bytes = encode_icns([
        (b"icp4", png_16),
        (b"icp5", png_32),
        (b"ic07", png_128),
        (b"ic08", png_256),
        (b"ic09", png_512),
    ])

    outputs = {
        "32x32.png": png_32,
        "128x128.png": png_128,
        "180x180.png": png_180,
        "192x192.png": png_192,
        "256x256.png": png_256,
        "512x512.png": png_512,
        "512x512-maskable.png": png_512_maskable,
        "og-image.png": png_og,
        "icon.ico": ico_bytes,
        "icon.icns": icns_bytes,
    }

    for filename, data in outputs.items():
        (icons_dir / filename).write_bytes(data)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    generate_all(repo_root / "math_explorer_gui" / "icons")


if __name__ == "__main__":
    main()
