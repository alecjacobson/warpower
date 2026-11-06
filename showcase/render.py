"""Render the saved scene with explicit per-process OptiX device selection."""

import sys
from pathlib import Path

import bpy

out = Path(__file__).resolve().parents[1] / "results/local/showcase"
preferences = bpy.context.preferences.addons["cycles"].preferences
preferences.compute_device_type = "OPTIX"
preferences.get_devices()
for device in preferences.devices:
    device.use = device.type == "OPTIX"
assert any(d.use for d in preferences.devices), "No OptiX device available"
scene = bpy.context.scene
scene.cycles.device = "GPU"
print("RENDER_DEVICES", [(d.name, d.type, d.use) for d in preferences.devices], flush=True)
if "--preview" in sys.argv:
    scene.render.image_settings.file_format = "PNG"
    for frame in [48, 180, 336, 540]:
        scene.frame_set(frame)
        scene.render.filepath = str(out / f"check-{frame}.png")
        bpy.ops.render.render(write_still=True)
else:
    scene.render.filepath = str(out / "render.mp4")
    bpy.ops.render.render(animation=True)
