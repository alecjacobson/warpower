"""Run with Blender 4.5: blender -b -t 4 --python showcase/build_scene.py."""

import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/local/showcase"
FPS = 24
LAST = FPS * 24
HEIGHT = 3.15


def key(obj, path, frame, value):
    setattr(obj, path, value)
    obj.keyframe_insert(data_path=path, frame=frame)


def material(name, color, metallic=0.0, roughness=0.25, emission=0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Coat Weight"].default_value = 0.35
    bsdf.inputs["Coat Roughness"].default_value = 0.18
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*color, 1)
        bsdf.inputs["Emission Strength"].default_value = emission
    return mat


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def area(name, position, color, power, size, target=(0, 0, HEIGHT)):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = power
    data.color = color
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    aim(obj, target)
    return obj


def main():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.get_devices()
    for device in preferences.devices:
        device.use = device.type == "OPTIX"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPTIX"
    scene.cycles.preview_samples = 16
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 4
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.035
    scene.render.use_persistent_data = True
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = LAST
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -0.35
    scene.world.color = (0.03, 0.03, 0.03)
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.065, 0.09, 0.16, 1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.4

    geometry = json.loads((OUT / "geometry.json").read_text())
    palette = [
        (0.025, 0.33, 0.38),
        (0.04, 0.49, 0.43),
        (0.06, 0.26, 0.54),
        (0.19, 0.12, 0.40),
        (0.40, 0.19, 0.48),
        (0.72, 0.32, 0.24),
        (0.76, 0.54, 0.24),
        (0.25, 0.53, 0.62),
    ]
    gems = [material(f"Ceramic / {j}", c, 0.27, 0.24) for j, c in enumerate(palette)]
    point_mat = material("Seed / warm white", (0.32, 0.85, 1.0), 0.2, 0.22, 2.3)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1)
    prototype = bpy.context.object
    sphere = prototype.data
    sphere.materials.append(point_mat)
    bpy.data.objects.remove(prototype, do_unlink=True)

    for item in geometry["cells"]:
        center = Vector(item["center"])
        mesh = bpy.data.meshes.new(f"Cell mesh {item['id']:03d}")
        mesh.from_pydata(item["vertices"], [], item["faces"])
        mesh.update()
        obj = bpy.data.objects.new(f"Power cell {item['id']:03d}", mesh)
        bpy.context.collection.objects.link(obj)
        # Broad spatial color bands, with a few warm inclusions.
        hue = (math.atan2(center.y, center.x) / (2 * math.pi) + 0.48) % 1.0
        index = int(hue * len(gems)) % len(gems)
        obj.data.materials.append(gems[index])
        bevel = obj.modifiers.new("Fine polished edges", "BEVEL")
        bevel.width = 0.012
        bevel.segments = 3
        bevel.affect = "EDGES"
        bevel.harden_normals = True
        obj.modifiers.new("Weighted facet normals", "WEIGHTED_NORMAL")
        start = 64 + int(center.length * 17) + item["id"] % 12
        for frame, scale in [
            (1, 0.001),
            (start, 0.001),
            (start + 40, 0.97),
            (216, 0.97),
            (312, 0.91),
            (390, 0.91),
            (480, 0.97),
            (LAST, 0.97),
        ]:
            key(obj, "scale", frame, (scale,) * 3)
        for frame, expansion in [
            (1, 1.0),
            (216, 1.0),
            (312, 1.47),
            (390, 1.47),
            (480, 1.0),
            (LAST, 1.0),
        ]:
            key(obj, "location", frame, center * expansion + Vector((0, 0, HEIGHT)))
        seed = bpy.data.objects.new(f"Seed {item['id']:03d}", sphere)
        bpy.context.collection.objects.link(seed)
        seed.location = Vector(item["seed"]) + Vector((0, 0, HEIGHT))
        for frame, scale in [
            (1, 0.001),
            (22 + item["id"] % 14, 0.035),
            (start + 12, 0.035),
            (start + 38, 0.001),
            (LAST, 0.001),
        ]:
            key(seed, "scale", frame, (scale,) * 3)

    # A restrained connection graph introduces the adjacency before cells appear.
    network = bpy.data.curves.new("Seed connections", "CURVE")
    network.dimensions = "3D"
    network.bevel_depth = 0.003
    network.bevel_resolution = 2
    seeds = [Vector(c["seed"]) for c in geometry["cells"]]
    for i, j in geometry["display_edges"]:
        spline = network.splines.new("POLY")
        spline.points.add(1)
        for point, coord in zip(spline.points, (seeds[i], seeds[j])):
            point.co = (*coord, 1)
    network_obj = bpy.data.objects.new("Connections", network)
    bpy.context.collection.objects.link(network_obj)
    network_obj.location.z = HEIGHT
    wire = bpy.data.materials.new("Connections / fading cyan")
    wire.use_nodes = True
    nodes = wire.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs[0].default_value = (0.025, 0.22, 0.30, 1)
    emission.inputs[1].default_value = 0.8
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    wire.node_tree.links.new(transparent.outputs[0], mix.inputs[1])
    wire.node_tree.links.new(emission.outputs[0], mix.inputs[2])
    wire.node_tree.links.new(mix.outputs[0], output.inputs[0])
    for frame, value in [(1, 0), (32, 1), (80, 1), (140, 0), (LAST, 0)]:
        mix.inputs[0].default_value = value
        mix.inputs[0].keyframe_insert(data_path="default_value", frame=frame)
    network.materials.append(wire)

    floor_mat = material("Midnight studio", (0.012, 0.019, 0.031), 0.32, 0.32)
    bpy.ops.mesh.primitive_plane_add(size=2000, location=(0, 0, -0.22))
    bpy.context.object.name = "Studio floor"
    bpy.context.object.data.materials.append(floor_mat)
    area("Key / silk", (0, -4, 9), (0.78, 0.9, 1.0), 1900, 6)
    area("Rim / turquoise", (-5, 1.5, 5.3), (0.19, 0.75, 1.0), 1500, 4)
    area("Rim / peach", (4, 3, 6), (1.0, 0.5, 0.26), 2100, 4)
    area("Front fill", (3, -5, 3), (0.8, 0.88, 1.0), 450, 5)

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera_data.lens = 49
    camera_data.clip_end = 3000
    camera_data.shift_x = -0.11
    camera_data.dof.use_dof = False
    # Constant-speed slow orbit; ease the radial pullback for the exploded view.
    for frame in range(1, LAST + 1, 8):
        t = (frame - 1) / (LAST - 1)
        angle = math.radians(-72 + 105 * t)
        expand = math.sin(math.pi * max(0, min(1, (frame - 200) / 290))) ** 2
        radius = 15.5 + 4.5 * expand
        camera.location = (radius * math.cos(angle), radius * math.sin(angle), 8.0)
        aim(camera, (0, 0, HEIGHT - 0.05))
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)
    for curve in camera.animation_data.action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"

    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    layer = nodes.new("CompositorNodeRLayers")
    glow = nodes.new("CompositorNodeGlare")
    glow.glare_type = "FOG_GLOW"
    glow.quality = "HIGH"
    glow.threshold = 1.2
    glow.size = 7
    glow.mix = -0.92
    composite = nodes.new("CompositorNodeComposite")
    scene.node_tree.links.new(layer.outputs["Image"], glow.inputs["Image"])
    scene.node_tree.links.new(glow.outputs["Image"], composite.inputs["Image"])
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.filepath = str(OUT / "render.mp4")
    scene.frame_set(245)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "warpower-showcase.blend"))
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(OUT / "preview.png")
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
