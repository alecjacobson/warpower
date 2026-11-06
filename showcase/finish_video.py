"""Create typography and an original ambient score, then finish the MP4."""

import argparse
import json
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/local/showcase"
FONTS = Path("/usr/share/fonts/truetype/dejavu")
WHITE = (233, 240, 245, 255)
MUTED = (134, 157, 175, 255)
CYAN = (99, 218, 221, 255)


def font(size, bold=False, mono=False):
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    return ImageFont.truetype(str(FONTS / (name + ("-Bold" if bold else "") + ".ttf")), size)


def tracked(draw, text, xy, size, spacing=3, color=WHITE):
    x, y = xy
    f = font(size)
    for char in text:
        draw.text((x, y), char, font=f, fill=color)
        x += draw.textlength(char, font=f) + spacing


def overlay(name, label=None, lines=None):
    canvas = Image.new("RGBA", (1920, 1080))
    draw = ImageDraw.Draw(canvas)
    if label:
        tracked(draw, label, (100, 328), 15, 2.3, CYAN)
    for i, line in enumerate(lines or []):
        draw.text((96, 379 + 78 * i), line, font=font(62, bold=True), fill=WHITE)
    canvas.save(OUT / f"{name}.png")
    return canvas, draw


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    header, draw = overlay("header")
    tracked(draw, "WARPOWER", (100, 65), 27, 7, WHITE)
    draw.line((100, 117, 140, 117), fill=CYAN, width=3)
    tracked(draw, "PURE NVIDIA WARP", (1460, 73), 14, 2, MUTED)
    draw.line((100, 971, 1820, 971), fill=(65, 92, 112, 120), width=1)
    draw.text((100, 997), "github.com/alecjacobson/warpower", font=font(16), fill=MUTED)
    draw.text((1480, 997), "176 WEIGHTED CELLS", font=font(15, mono=True), fill=MUTED)
    header.save(OUT / "header.png")
    intro, d = overlay("intro", "01  /  SEED POINTS", ["From points", "to power."])
    d.text((100, 573), "A three-dimensional partition of space.", font=font(22), fill=MUTED)
    intro.save(OUT / "intro.png")
    built, d = overlay("built", "02  /  POWER DIAGRAM", ["Space,", "partitioned."])
    d.text((100, 573), "Weighted sites. Convex cells.", font=font(22), fill=MUTED)
    built.save(OUT / "built.png")
    exploded, d = overlay("exploded", "03  /  EXPLODED VIEW", ["Every cell.", "Every facet."])
    d.text((100, 573), "Computed adjacency, revealed in 3D.", font=font(22), fill=MUTED)
    exploded.save(OUT / "exploded.png")
    card, d = overlay("benchmark", "04  /  MEASURED FLAGSHIP")
    d.text((92, 365), "2.17×", font=font(112, bold=True), fill=WHITE)
    d.text((100, 508), "faster than Paragram", font=font(28), fill=WHITE)
    d.line((100, 575, 490, 575), fill=(65, 92, 112, 150), width=1)
    d.text((100, 606), "30.4 ms  /  66.0 ms", font=font(29), fill=CYAN)
    d.text((100, 660), "100,000 clustered Voronoi sites", font=font(20), fill=MUTED)
    d.text((100, 693), "NVIDIA L40 · mean of 10 calls", font=font(20), fill=MUTED)
    card.save(OUT / "benchmark.png")

    # Original synthesized score: gently detuned pads and sparse bell overtones.
    rate = 48000
    t = np.arange(rate * 24) / rate
    signal = np.zeros((len(t), 2))
    rng = np.random.default_rng(31)
    for start, notes in [
        (0, [45, 52, 56, 59, 64]),
        (8, [38, 45, 52, 54, 61]),
        (16, [40, 47, 52, 56, 61]),
    ]:
        local = t - start
        env = np.clip(local / 2, 0, 1) * np.clip((11 - local) / 3, 0, 1)
        env = np.sin(np.pi * env / 2) ** 2
        for index, note in enumerate(notes):
            freq = 440 * 2 ** ((note - 69) / 12)
            phase = rng.uniform(0, 2 * np.pi)
            for channel, detune in enumerate([-0.07, 0.07]):
                phase_wave = (
                    2 * np.pi * (freq + detune) * t + phase + 0.1 * np.sin(2 * np.pi * 0.13 * t)
                )
                tone = (
                    np.sin(phase_wave)
                    + 0.18 * np.sin(2 * phase_wave)
                    + 0.045 * np.sin(3 * phase_wave)
                )
                signal[:, channel] += env * tone * (0.013 if index else 0.023)
    for start, note in [(3.4, 76), (5.8, 80), (10.1, 83), (13.0, 78), (18.6, 80), (20.4, 76)]:
        local = np.maximum(t - start, 0)
        env = (1 - np.exp(-local * 30)) * np.exp(-local * 0.9) * (t >= start)
        freq = 440 * 2 ** ((note - 69) / 12)
        bell = (
            0.018
            * env
            * (np.sin(2 * np.pi * freq * local) + 0.15 * np.sin(2 * np.pi * freq * 2.004 * local))
        )
        signal[:, 0] += bell
        signal[:, 1] += np.roll(bell, int(rate * 0.018)) * 0.85
    signal *= (np.clip(t / 1.5, 0, 1) * np.clip((24 - t) / 1.8, 0, 1))[:, None]
    signal = np.tanh(signal * 2.5)
    with wave.open(str(OUT / "score.wav"), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((signal * 32767).astype("<i2").tobytes())
    print("Prepared five typography layers and original 24-second stereo score.")


def finish():
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-hide_banner", "-y", "-i", str(OUT / "render.mp4")]
    for name in ["header", "intro", "built", "exploded", "benchmark"]:
        cmd += ["-loop", "1", "-framerate", "24", "-i", str(OUT / f"{name}.png")]
    cmd += ["-i", str(OUT / "score.wav")]
    filters = ["[0:v][1:v]overlay=0:0[v1]"]
    for i, (start, end) in enumerate([(0.7, 5.6), (6.1, 9.7), (10.2, 18.0), (18.6, 24.5)], 2):
        filters += [
            f"[{i}:v]format=rgba,fade=t=in:st={start}:d=0.55:alpha=1,"
            f"fade=t=out:st={end}:d=0.45:alpha=1[t{i}]",
            f"[v{i - 1}][t{i}]overlay=0:0[v{i}]",
        ]
    filters += ["[v5]fade=t=in:st=0:d=0.7,fade=t=out:st=23.2:d=0.8,format=yuv420p[final]"]
    cmd += [
        "-filter_complex_threads",
        "2",
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[final]",
        "-map",
        "6:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-threads",
        "4",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        "24",
        "-movflags",
        "+faststart",
        "-metadata",
        "title=Warpower — From points to power",
        "-metadata",
        "comment=Actual Warpower adjacency; offline Blender render. Original synthesized score.",
        str(OUT / "warpower-showcase.mp4"),
    ]
    subprocess.run(cmd, check=True)
    (OUT / "encoding-command.json").write_text(json.dumps(cmd, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finish", action="store_true")
    args = parser.parse_args()
    prepare()
    if args.finish:
        finish()
