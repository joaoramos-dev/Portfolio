"""Repair selected OfficeAssets model issues and export a portfolio GLB.

Run through Blender:
  blender --background OfficeAssets_free.blend --python repair_office_model.py -- \
    --blend-output OfficeAssets_portfolio.blend \
    --glb-output public/models/desk_room_fixed.glb

The source .blend is read-only. The monitor and chair are composed for the
front-facing portfolio view, cactus flowers are added, and alpha blending for
cutout-only materials is converted to glTF MASK to avoid depth sorting errors.
"""

from __future__ import annotations

import argparse
import bmesh
import json
import math
import struct
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


FLOWER_PETAL = "Portfolio_Cactus_Flower_Petal"
FLOWER_CENTER = "Portfolio_Cactus_Flower_Center"
FLOWER_PREFIX = "Portfolio_Cactus_Flower_"
MUG_PREFIX = "Portfolio_Coffee_Mug"


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend-output", required=True)
    parser.add_argument("--glb-output", required=True)
    return parser.parse_args(argv)


def get_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    principled = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.62
    return material


def make_wood_albedo_less_red(source: bpy.types.Image, output_path: Path) -> bpy.types.Image:
    """Keep the wood grain while shifting its dark red tint toward warm brown."""
    width, height = source.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    source.pixels.foreach_get(pixels)
    pixels = pixels.reshape((-1, 4))
    pixels[:, 0] *= 0.82
    pixels[:, 1] *= 1.3
    pixels[:, 2] *= 1.8
    np.clip(pixels[:, :3], 0.0, 1.0, out=pixels[:, :3])

    image = bpy.data.images.new("Portfolio_Tabletop_Wood_Albedo", width, height, alpha=False)
    image.colorspace_settings.name = "sRGB"
    image.pixels.foreach_set(pixels.ravel())
    image.filepath_raw = str(output_path)
    image.file_format = "JPEG"
    image.save()
    bpy.data.images.remove(image)

    image = bpy.data.images.load(str(output_path), check_existing=False)
    image.colorspace_settings.name = "sRGB"
    image.pack()
    return image


def append_ellipsoid(vertices, faces, face_materials, center, radii, material_index):
    segments, rings = 12, 6
    base = len(vertices)
    for ring in range(rings + 1):
        phi = math.pi * ring / rings
        for segment in range(segments):
            theta = 2 * math.pi * segment / segments
            vertices.append((
                center[0] + radii[0] * math.sin(phi) * math.cos(theta),
                center[1] + radii[1] * math.sin(phi) * math.sin(theta),
                center[2] + radii[2] * math.cos(phi),
            ))
    for ring in range(rings):
        for segment in range(segments):
            a = base + ring * segments + segment
            b = base + ring * segments + (segment + 1) % segments
            c = base + (ring + 1) * segments + (segment + 1) % segments
            d = base + (ring + 1) * segments + segment
            faces.extend(((a, b, c), (a, c, d)))
            face_materials.extend((material_index, material_index))


def add_cactus_flowers():
    cactus = bpy.data.objects.get("Cactus")
    if not cactus or cactus.type != "MESH":
        raise RuntimeError("Expected Cactus mesh in source scene")

    for obj in list(bpy.data.objects):
        if obj.name.startswith(FLOWER_PREFIX):
            bpy.data.objects.remove(obj, do_unlink=True)

    petals = get_material(FLOWER_PETAL, (0.96, 0.27, 0.33, 1.0))
    center_mat = get_material(FLOWER_CENTER, (1.0, 0.68, 0.16, 1.0))

    # Local coordinates follow the cactus mesh's upright Z axis. Each bloom
    # sits just above an existing cluster tip and stays small relative to it.
    tips = [
        (0.015, 0.064, 0.428),
        (-0.036, 0.066, 0.397),
        (0.113, 0.158, 0.355),
        (-0.081, -0.003, 0.341),
    ]
    for index, (x, y, z) in enumerate(tips, start=1):
        vertices, faces, material_ids = [], [], []
        for petal in range(5):
            angle = math.tau * petal / 5
            radial = (math.cos(angle), math.sin(angle))
            tangent = (-radial[1], radial[0])
            origin = (x + radial[0] * 0.009, y + radial[1] * 0.009, z + 0.004)
            start = len(vertices)
            append_ellipsoid(vertices, faces, material_ids, origin, (0.011, 0.006, 0.0035), 0)
            # Rotate each petal's local long axis from X into the radial axis.
            for vertex_index in range(start, len(vertices)):
                vx, vy, vz = vertices[vertex_index]
                dx, dy = vx - origin[0], vy - origin[1]
                vertices[vertex_index] = (
                    origin[0] + radial[0] * dx + tangent[0] * dy,
                    origin[1] + radial[1] * dx + tangent[1] * dy,
                    vz,
                )
        append_ellipsoid(vertices, faces, material_ids, (x, y, z + 0.008), (0.006, 0.006, 0.0045), 1)

        mesh = bpy.data.meshes.new(f"{FLOWER_PREFIX}{index:02d}_Mesh")
        mesh.from_pydata(vertices, [], faces)
        mesh.materials.append(petals)
        mesh.materials.append(center_mat)
        for polygon, material_id in zip(mesh.polygons, material_ids):
            polygon.material_index = material_id
            polygon.use_smooth = True
        flower = bpy.data.objects.new(f"{FLOWER_PREFIX}{index:02d}", mesh)
        bpy.context.scene.collection.objects.link(flower)
        flower.parent = cactus
        flower.matrix_parent_inverse = Matrix.Identity(4)
        flower.location = (0, 0, 0)


def add_coffee_mug():
    # The frame is centered at Blender Y=-1.116. Put the mug 22 cm behind it
    # on the desk so it reads as a neighboring personal object from the POV.
    mug_origin = (-0.034, -1.436, 0.743)
    body_name = f"{MUG_PREFIX}_Body"
    body = bpy.data.objects.get(body_name)
    if body is None:
        ceramic = get_material("Portfolio_Coffee_Mug_Ceramic", (0.43, 0.32, 0.22, 1.0))
        principled = next(node for node in ceramic.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
        principled.inputs["Roughness"].default_value = 0.3
        principled.inputs["Coat Weight"].default_value = 0.12
        principled.inputs["Coat Roughness"].default_value = 0.24

        # Lathed, hollow ceramic wall with a softly rounded rim and a closed inner base.
        segments = 48
        profile = [
            (0.040, 0.000), (0.044, 0.005), (0.050, 0.081),
            (0.050, 0.088), (0.044, 0.088), (0.044, 0.014),
        ]
        vertices, faces = [], []
        for radius, height in profile:
            for segment in range(segments):
                angle = math.tau * segment / segments
                vertices.append((radius * math.cos(angle), radius * math.sin(angle), height))
        for ring in range(len(profile) - 1):
            lower = ring * segments
            upper = (ring + 1) * segments
            for segment in range(segments):
                nxt = (segment + 1) % segments
                faces.append((lower + segment, lower + nxt, upper + nxt, upper + segment))
        bottom_center = len(vertices)
        vertices.append((0.0, 0.0, 0.0))
        inner_center = len(vertices)
        vertices.append((0.0, 0.0, 0.014))
        inner_ring = (len(profile) - 1) * segments
        for segment in range(segments):
            nxt = (segment + 1) % segments
            faces.append((bottom_center, nxt, segment))
            faces.append((inner_center, inner_ring + segment, inner_ring + nxt))

        body_mesh = bpy.data.meshes.new(f"{MUG_PREFIX}_Body_Mesh")
        body_mesh.from_pydata(vertices, [], faces)
        body_mesh.materials.append(ceramic)
        for polygon in body_mesh.polygons:
            polygon.use_smooth = True
        body = bpy.data.objects.new(body_name, body_mesh)
        bpy.context.scene.collection.objects.link(body)
        body.location = mug_origin
    else:
        ceramic = body.data.materials[0]

    if bpy.data.objects.get(f"{MUG_PREFIX}_Coffee") is None:
        coffee = get_material("Portfolio_Coffee_Surface", (0.055, 0.018, 0.008, 1.0))
        coffee_principled = next(node for node in coffee.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
        coffee_principled.inputs["Roughness"].default_value = 0.18
        coffee_principled.inputs["Coat Weight"].default_value = 0.25
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=48,
            radius=0.041,
            depth=0.002,
            location=(mug_origin[0], mug_origin[1], mug_origin[2] + 0.075),
        )
        surface = bpy.context.object
        surface.name = f"{MUG_PREFIX}_Coffee"
        surface.data.name = f"{MUG_PREFIX}_Coffee_Mesh"
        surface.data.materials.append(coffee)

    old_handle = bpy.data.objects.get(f"{MUG_PREFIX}_Handle")
    if old_handle:
        old_mesh = old_handle.data
        bpy.data.objects.remove(old_handle, do_unlink=True)
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)

    # The seated camera looks along Blender +X. Negative Y maps to screen-left
    # in the portfolio's front view, so place the lateral handle on that side.
    # A round half-circle handle opens toward the cup; both ends embed into
    # its side wall and the outer arc stays proportional to the cup height.
    handle_side = -1.0
    arc_center_y = handle_side * 0.0445
    arc_outer_y = handle_side * 0.0675
    arc_center_z = 0.047
    arc_radius = 0.023
    arc_control = arc_radius * 0.55228475
    bezier_segments = [
        (
            (0.0, arc_center_y, arc_center_z + arc_radius),
            (0.0, arc_center_y + handle_side * arc_control, arc_center_z + arc_radius),
            (0.0, arc_outer_y, arc_center_z + arc_control),
            (0.0, arc_outer_y, arc_center_z),
        ),
        (
            (0.0, arc_outer_y, arc_center_z),
            (0.0, arc_outer_y, arc_center_z - arc_control),
            (0.0, arc_center_y + handle_side * arc_control, arc_center_z - arc_radius),
            (0.0, arc_center_y, arc_center_z - arc_radius),
        ),
    ]
    path = []
    samples_per_curve = 24
    for curve_index, (p0, p1, p2, p3) in enumerate(bezier_segments):
        first_sample = 0 if curve_index == 0 else 1
        for sample in range(first_sample, samples_per_curve + 1):
            t = sample / samples_per_curve
            u = 1.0 - t
            point = tuple(
                u**3 * p0[axis] + 3 * u**2 * t * p1[axis]
                + 3 * u * t**2 * p2[axis] + t**3 * p3[axis]
                for axis in range(3)
            )
            path.append(point)

    handle_vertices, handle_faces = [], []
    tube_radius, tube_sides = 0.0048, 12
    for index, point in enumerate(path):
        before = Vector(path[max(index - 1, 0)])
        after = Vector(path[min(index + 1, len(path) - 1)])
        tangent = (after - before).normalized()
        in_plane_normal = Vector((1.0, 0.0, 0.0)).cross(tangent).normalized()
        end_distance = min(index, len(path) - 1 - index)
        root_fade = max(0.0, 1.0 - end_distance / (samples_per_curve * 0.55))
        local_radius = tube_radius + 0.0032 * root_fade
        for side in range(tube_sides):
            angle = math.tau * side / tube_sides
            offset = Vector((1.0, 0.0, 0.0)) * (math.cos(angle) * local_radius)
            offset += in_plane_normal * (math.sin(angle) * local_radius)
            handle_vertices.append(tuple(Vector(point) + offset))

    for ring in range(len(path) - 1):
        for side in range(tube_sides):
            nxt = (side + 1) % tube_sides
            a = ring * tube_sides + side
            b = ring * tube_sides + nxt
            c = (ring + 1) * tube_sides + nxt
            d = (ring + 1) * tube_sides + side
            handle_faces.append((a, b, c, d))
    handle_faces.append(tuple(range(tube_sides - 1, -1, -1)))
    end_start = (len(path) - 1) * tube_sides
    handle_faces.append(tuple(end_start + side for side in range(tube_sides)))

    handle_mesh = bpy.data.meshes.new(f"{MUG_PREFIX}_Handle_Mesh")
    handle_mesh.from_pydata(handle_vertices, [], handle_faces)
    handle_mesh.materials.append(ceramic)
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(handle_mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(handle_mesh)
    bm.free()
    for polygon in handle_mesh.polygons:
        polygon.use_smooth = True
    handle = bpy.data.objects.new(f"{MUG_PREFIX}_Handle", handle_mesh)
    bpy.context.scene.collection.objects.link(handle)
    handle.location = mug_origin


def make_lamp_material(name, color, roughness, metallic=0.0, transmission=0.0, emission=None, emission_strength=0.0):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1.0)
    material.use_nodes = True
    shader = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Transmission Weight"].default_value = transmission
    if emission:
        shader.inputs["Emission Color"].default_value = (*emission, 1.0)
        shader.inputs["Emission Strength"].default_value = emission_strength
    return material


def create_lamp_lathe(shade, name, profile, material, segments=32, cap_start=True, cap_end=True):
    vertices, faces = [], []
    for x, radius in profile:
        for segment in range(segments):
            angle = math.tau * segment / segments
            vertices.append((x, radius * math.cos(angle), radius * math.sin(angle)))
    for ring in range(len(profile) - 1):
        lower, upper = ring * segments, (ring + 1) * segments
        for segment in range(segments):
            nxt = (segment + 1) % segments
            faces.append((lower + segment, lower + nxt, upper + nxt, upper + segment))
    if cap_start:
        faces.append(tuple(reversed(range(segments))))
    if cap_end:
        faces.append(tuple((len(profile) - 1) * segments + i for i in range(segments)))

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.use_smooth = len(polygon.vertices) == 4
    mesh.transform(shade.matrix_world)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def create_lamp_filament(shade, material):
    paths = [
        ((0.034, -0.005, -0.005), (0.04, -0.006, -0.005), (0.046, 0.005, -0.004),
         (0.052, -0.005, -0.003), (0.058, 0.005, -0.002), (0.064, 0.0, 0.0)),
        ((0.028, -0.009, -0.007), (0.042, -0.009, -0.007)),
        ((0.028, 0.009, 0.007), (0.042, 0.009, 0.007)),
    ]
    vertices, faces = [], []
    sides, radius = 6, 0.0009
    for path in paths:
        for start, end in zip(path, path[1:]):
            a, b = Vector(start), Vector(end)
            direction = (b - a).normalized()
            tangent = direction.cross(Vector((0, 0, 1)))
            if tangent.length < 1e-5:
                tangent = direction.cross(Vector((0, 1, 0)))
            tangent.normalize()
            bitangent = direction.cross(tangent).normalized()
            base = len(vertices)
            for point in (a, b):
                for side in range(sides):
                    angle = math.tau * side / sides
                    vertex = point + radius * (math.cos(angle) * tangent + math.sin(angle) * bitangent)
                    vertices.append(tuple(vertex))
            for side in range(sides):
                nxt = (side + 1) % sides
                faces.append((base + side, base + nxt, base + sides + nxt, base + sides + side))
            faces.append(tuple(base + side for side in reversed(range(sides))))
            faces.append(tuple(base + sides + side for side in range(sides)))

    mesh = bpy.data.meshes.new("Portfolio_Lamp_Filament_Mesh")
    mesh.from_pydata(vertices, [], faces)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.transform(shade.matrix_world)
    mesh.materials.append(material)
    obj = bpy.data.objects.new("Portfolio_Lamp_Filament", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def add_real_lamp_bulb(shade):
    for name in ("Portfolio_Lamp_Bulb_Glass", "Portfolio_Lamp_Bulb_Base", "Portfolio_Lamp_Filament"):
        old = bpy.data.objects.get(name)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)

    glass = make_lamp_material(
        "Portfolio_Lamp_Bulb_Glass_Material",
        (0.96, 0.88, 0.72), 0.08, transmission=0.92,
    )
    socket = make_lamp_material("Portfolio_Lamp_Socket_Material", (0.28, 0.25, 0.20), 0.28, metallic=0.82)
    filament = make_lamp_material(
        "Portfolio_Lamp_Filament_Material",
        (0.9, 0.22, 0.025), 0.3, emission=(1.0, 0.19, 0.015), emission_strength=1.6,
    )
    create_lamp_lathe(
        shade, "Portfolio_Lamp_Bulb_Glass",
        [
            (0.018, 0.008), (0.026, 0.011), (0.034, 0.017),
            (0.046, 0.023), (0.060, 0.026), (0.073, 0.023),
            (0.084, 0.017), (0.092, 0.009), (0.096, 0.001),
        ],
        glass, cap_start=False,
    )
    create_lamp_lathe(
        shade, "Portfolio_Lamp_Bulb_Base",
        [
            (-0.006, 0.0072), (-0.005, 0.0090), (-0.0038, 0.0095),
            (-0.0025, 0.0085), (-0.0008, 0.0085), (0.0005, 0.0095),
            (0.0017, 0.0085), (0.0034, 0.0085), (0.0047, 0.0095),
            (0.0059, 0.0085), (0.0076, 0.0085), (0.0089, 0.0095),
            (0.0101, 0.0085), (0.0118, 0.0085), (0.0131, 0.0095),
            (0.0143, 0.0085), (0.016, 0.0075), (0.018, 0.008),
        ],
        socket, segments=24, cap_end=False,
    )
    create_lamp_filament(shade, filament)


def add_portfolio_lighting_rig():
    """Create the Blender look-dev camera and a 40-second animated desk-lamp spot."""
    from bpy_extras import anim_utils

    scene = bpy.context.scene
    scene.render.fps = 25
    scene.frame_start = 1
    scene.frame_end = 1001
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    camera_data = bpy.data.cameras.get("Portfolio_Lookdev_Camera")
    camera = bpy.data.objects.get("Portfolio_Lookdev_Camera")
    if camera is None:
        camera_data = camera_data or bpy.data.cameras.new("Portfolio_Lookdev_Camera")
        camera = bpy.data.objects.new("Portfolio_Lookdev_Camera", camera_data)
        scene.collection.objects.link(camera)
    camera.location = (3.15, -1.15, 1.75)
    camera.rotation_euler = (Vector((0.18, -1.16, 0.88)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 32.5
    scene.camera = camera

    shade = bpy.data.objects.get("Lamp_Shade.001")
    if shade is None:
        raise RuntimeError("Expected the desk lamp shade in the Blender source scene")
    bpy.context.view_layer.update()
    add_real_lamp_bulb(shade)

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("Portfolio_Noir_World")
        scene.world = world
    world.use_nodes = True
    background = next((node for node in world.node_tree.nodes if node.type == "BACKGROUND"), None)
    if background is None:
        background = world.node_tree.nodes.new("ShaderNodeBackground")
        output = next((node for node in world.node_tree.nodes if node.type == "OUTPUT_WORLD"), None)
        if output is None:
            output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
        world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
    background.inputs["Color"].default_value = (0.12, 0.085, 0.055, 1.0)
    background.inputs["Strength"].default_value = 0.12

    lamp_data = bpy.data.lights.get("Portfolio_Desk_Lamp")
    if lamp_data is None:
        lamp_data = bpy.data.lights.new("Portfolio_Desk_Lamp", type="SPOT")
    lamp_data.energy = 90.0
    lamp_data.color = (1.0, 0.64, 0.36)
    lamp_data.spot_size = math.radians(94)
    lamp_data.spot_blend = 0.78
    lamp_data.shadow_soft_size = 0.045
    lamp_data.use_shadow = True
    lamp = bpy.data.objects.get("Portfolio_Desk_Lamp")
    if lamp is None:
        lamp = bpy.data.objects.new("Portfolio_Desk_Lamp", lamp_data)
        scene.collection.objects.link(lamp)
    else:
        lamp.data = lamp_data
    emitter_position = shade.matrix_world @ Vector((0.05, 0.0, 0.0))
    lamp.location = emitter_position
    beam_direction = Vector((0.1044, 0.4789, -0.8716)).normalized()
    target = Vector(lamp.location) + beam_direction * 1.15
    lamp.rotation_euler = beam_direction.to_track_quat("-Z", "Y").to_euler()

    flicker_keys = [
        (0.0, 0.08), (0.12, 0.82), (0.18, 0.16), (0.23, 0.76),
        (0.27, 0.05), (0.4, 0.9), (0.47, 0.48), (0.54, 1.0), (0.7, 1.0),
        (10.25, 0.92), (10.32, 0.08), (10.44, 1.0), (10.51, 0.5), (10.61, 0.96),
        (20.3, 0.95), (20.38, 0.05), (20.5, 1.0), (20.58, 0.28), (20.7, 0.94),
        (30.4, 0.95), (30.47, 0.06), (30.59, 0.9), (30.67, 0.35), (30.8, 0.95),
        (39.5, 0.95), (39.56, 0.04), (39.68, 0.9), (39.77, 0.25), (39.86, 0.95),
        (39.94, 0.08), (40.0, 0.08),
    ]
    lamp_data.energy = 90.0 * flicker_keys[0][1]
    lamp_data.keyframe_insert(data_path="energy", frame=scene.frame_start, group="Lamp Flicker")
    action = lamp_data.animation_data.action
    action.name = "Portfolio_Lamp_Flicker_40s"
    slot = lamp_data.animation_data.action_slot
    channelbag = anim_utils.action_ensure_channelbag_for_slot(action, slot)
    energy_curve = channelbag.fcurves.ensure("energy", index=0, group_name="Lamp Flicker")
    energy_curve.keyframe_points.clear()
    for seconds, multiplier in flicker_keys:
        point = energy_curve.keyframe_points.insert(
            scene.frame_start + seconds * scene.render.fps,
            90.0 * multiplier,
        )
        point.interpolation = "LINEAR"
    energy_curve.modifiers.new(type="CYCLES")

    scene.frame_end = scene.frame_start + 40 * scene.render.fps - 1
    scene.frame_set(scene.frame_start)
    print("Created Blender look-dev camera and 40-second lamp flicker action")


def fix_monitor_shell_shading():
    """Remove weighted normals that create triangular shading artifacts on the CRT shell."""
    monitor = bpy.data.objects.get("Monitor")
    if monitor is None or monitor.type != "MESH":
        raise RuntimeError("Expected Monitor mesh in source scene")

    weighted_normals = [
        modifier for modifier in monitor.modifiers
        if modifier.type == "WEIGHTED_NORMAL"
    ]
    for modifier in weighted_normals:
        monitor.modifiers.remove(modifier)
    if weighted_normals:
        print("Removed Weighted Normal from the monitor shell")


def compose_front_view():
    monitor = bpy.data.objects.get("Monitor")
    chair = bpy.data.objects.get("Chair")
    computer = bpy.data.objects.get("Computer")
    pinboard = bpy.data.objects.get("PinBoard")
    lamp = bpy.data.objects.get("Lamp01")
    keyboard = bpy.data.objects.get("Keyboard")
    mouse = bpy.data.objects.get("Computer_Mouse")
    if not all((monitor, computer, chair, pinboard, lamp, keyboard, mouse)):
        raise RuntimeError("Expected desk, monitor, keyboard, chair, pinboard, and lamp meshes")

    # The source CRT already faces the room's front (+X) side. Keep its
    # authored rotation; adding pi points the display away from the camera.

    # Center the chair on the desk so its backrest frames the foreground view.
    # Blender Y maps to negative glTF Z in this asset's coordinate conversion.
    chair.location.y = -1.17
    chair.scale.x *= 1.05
    chair.scale.y *= 1.10
    chair.scale.z *= 1.05

    # Match the reference's larger corkboard while keeping its frame centered.
    # Blender Z is the glTF vertical axis.
    pinboard.scale.y *= 1.30
    pinboard.scale.z *= 1.30
    pinboard.location.z -= 0.01

    # Enlarge the CRT vertically from the tabletop and shift the typing area
    # left so the keyboard sits near the desk's center, as in the reference.
    desk_pivot = Vector((0.10, -0.49, 0.743))
    scale_matrix = Matrix.Translation(desk_pivot) @ Matrix.Diagonal((1.0, 1.0, 1.16, 1.0)) @ Matrix.Translation(-desk_pivot)
    for obj in (computer, monitor):
        obj.matrix_world = scale_matrix @ obj.matrix_world

    keyboard.location.y -= 0.20
    mouse.location.y -= 0.14

    # The articulated arm is larger in the reference and its base sits farther
    # left. Scaling the root carries every linked arm segment with it.
    lamp.scale = tuple(component * 1.30 for component in lamp.scale)
    lamp.location.y -= 0.14

    # Move the cigarette case and ashtray cluster left of the keyboard.
    for name in (
        "Ashtray", "Ashtray_Ash", "Cigarette_Normal", "Cigarette_Stub_Bent",
        "Cigarette_Stub_Straight", "CigarettePack_Blue", "CigarettePack_Brown",
    ):
        prop = bpy.data.objects.get(name)
        if prop:
            prop.location.y -= 0.25

    # The reference uses three deliberate pieces of paper, recreated as a
    # lit canvas overlay in the web scene rather than the asset's stock atlas.
    for name in (
        "Paper_Fax1", "Paper_Fax2", "Paper_Note", "Photo_Child", "Photo_Family",
        "Pin_Big1", "Pin_Big2", "Pin_Big2.001", "Pin_Flat1", "Pin_Flat1.001", "Pin_Flat2",
    ):
        obj = bpy.data.objects.get(name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)


def improve_table_shading():
    table = bpy.data.objects.get("Table_Rough")
    if not table or table.type != "MESH":
        raise RuntimeError("Expected Table_Rough mesh in source scene")

    # The legacy atlas maps are removed separately below; keep mesh shading
    # smooth so bevel highlights do not read as extra streaks.
    for material in table.data.materials:
        if not material or not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type == "NORMAL_MAP":
                node.inputs["Strength"].default_value = 0.32

    # Preserve the tabletop's broad flat planes while smoothing its small bevels.
    mesh = table.data
    edge_faces = {tuple(sorted(edge.vertices)): [] for edge in mesh.edges}
    for polygon in mesh.polygons:
        vertices = list(polygon.vertices)
        for index, vertex in enumerate(vertices):
            edge = tuple(sorted((vertex, vertices[(index + 1) % len(vertices)])))
            if edge in edge_faces:
                edge_faces[edge].append(polygon)
    hard_angle = math.radians(38)
    for edge in mesh.edges:
        linked = edge_faces[tuple(sorted(edge.vertices))]
        if len(linked) == 2:
            edge.use_edge_sharp = linked[0].normal.angle(linked[1].normal) > hard_angle
    for polygon in mesh.polygons:
        polygon.use_smooth = True

    weighted = next((modifier for modifier in table.modifiers if modifier.type == "WEIGHTED_NORMAL"), None)
    if weighted is None:
        weighted = table.modifiers.new("Portfolio_Table_WeightedNormals", "WEIGHTED_NORMAL")
    weighted.keep_sharp = True
    weighted.weight = 50
    weighted.thresh = 0.01

    bpy.context.view_layer.objects.active = table
    table.select_set(True)
    bpy.ops.object.modifier_apply(modifier=weighted.name)
    table.select_set(False)


def remove_legacy_table_grain():
    """Flatten the source atlas material still assigned to non-top table faces."""
    table = bpy.data.objects.get("Table_Rough")
    if not table or table.type != "MESH":
        raise RuntimeError("Expected Table_Rough mesh in source scene")

    # M_Table_2048 is the model's original atlas. Its color and normal maps
    # contain strong painted streaks that remain visible on the table apron
    # and rails even after the broad tabletop gets its replacement material.
    material = bpy.data.materials.get("M_Table_2048")
    if not material or not material.use_nodes:
        return
    nodes = material.node_tree.nodes
    for node in list(nodes):
        if node.type in {"TEX_IMAGE", "NORMAL_MAP", "BUMP"}:
            nodes.remove(node)
    principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if principled is None:
        principled = nodes.new("ShaderNodeBsdfPrincipled")
        output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)
        if output is None:
            output = nodes.new("ShaderNodeOutputMaterial")
        material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    principled.inputs["Base Color"].default_value = (0.075, 0.055, 0.04, 1.0)
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.94


def apply_real_tabletop_material():
    table = bpy.data.objects["Table_Rough"]
    material = bpy.data.materials.get("Portfolio_Tabletop_Wood") or bpy.data.materials.new("Portfolio_Tabletop_Wood")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (620, 80)
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (340, 80)
    shader.inputs["Metallic"].default_value = 0.0
    shader.inputs["Roughness"].default_value = 0.94
    shader.inputs["Coat Weight"].default_value = 0.0
    shader.inputs["Coat Roughness"].default_value = 0.8
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    # Use the dark, linear-grain wood set already bundled with the web project.
    # Its long grain reads closer to the reference desk under the warm lamp.
    texture_root = Path(__file__).resolve().parents[1] / "public" / "textures" / "wood_table_001"
    texture_specs = (
        ("wood_table_001_diff_2k.jpg", "sRGB"),
        ("wood_table_001_rough_2k.jpg", "Non-Color"),
        ("wood_table_001_nor_gl_2k.jpg", "Non-Color"),
    )
    images = {}
    for filename, color_space in texture_specs:
        image_path = texture_root / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing tabletop texture: {image_path}")
        image = bpy.data.images.load(str(image_path), check_existing=True)
        image.colorspace_settings.name = color_space
        image.pack()
        images[filename] = image

    uv_map = nodes.new("ShaderNodeUVMap")
    uv_map.uv_map = "Portfolio_TableTop_UV"
    color_texture = nodes.new("ShaderNodeTexImage")
    color_texture.image = make_wood_albedo_less_red(
        images["wood_table_001_diff_2k.jpg"],
        texture_root / "wood_table_001_diff_portfolio.jpg",
    )
    color_texture.interpolation = "Linear"
    roughness_texture = nodes.new("ShaderNodeTexImage")
    roughness_texture.image = images["wood_table_001_rough_2k.jpg"]
    roughness_texture.interpolation = "Linear"
    normal_texture = nodes.new("ShaderNodeTexImage")
    normal_texture.image = images["wood_table_001_nor_gl_2k.jpg"]
    normal_texture.interpolation = "Linear"
    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.inputs["Strength"].default_value = 0.08
    links.new(uv_map.outputs["UV"], color_texture.inputs["Vector"])
    links.new(uv_map.outputs["UV"], roughness_texture.inputs["Vector"])
    links.new(uv_map.outputs["UV"], normal_texture.inputs["Vector"])
    links.new(color_texture.outputs["Color"], shader.inputs["Base Color"])
    links.new(roughness_texture.outputs["Color"], shader.inputs["Roughness"])
    links.new(normal_texture.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])

    mesh = table.data
    uv_layer = mesh.uv_layers.get("Portfolio_TableTop_UV") or mesh.uv_layers.new(name="Portfolio_TableTop_UV")
    original_uv = next((layer for layer in mesh.uv_layers if layer != uv_layer), None)
    if original_uv:
        for loop in mesh.loops:
            uv_layer.data[loop.index].uv = original_uv.data[loop.index].uv
    xs = [vertex.co.x for vertex in mesh.vertices]
    ys = [vertex.co.y for vertex in mesh.vertices]
    min_x, min_y = min(xs), min(ys)
    span_x, span_y = max(xs) - min_x, max(ys) - min_y
    # Use a single, non-repeating tile over the entire top. U follows the
    # tabletop's long Y axis; V spans its width so repeat seams cannot show.
    tabletop_left_u = 0.0
    material_index = next((i for i, existing in enumerate(mesh.materials) if existing == material), None)
    if material_index is None:
        mesh.materials.append(material)
        material_index = len(mesh.materials) - 1
    top_z = max(vertex.co.z for vertex in mesh.vertices)
    for polygon in mesh.polygons:
        is_tabletop = polygon.center.z >= top_z - 0.065
        if is_tabletop:
            polygon.material_index = material_index
            for loop_index in polygon.loop_indices:
                vertex = mesh.vertices[mesh.loops[loop_index].vertex_index]
                uv_layer.data[loop_index].uv = (
                    tabletop_left_u + (vertex.co.x - min_x) / span_x,
                    (vertex.co.y - min_y) / span_y,
                )


def make_desk_lamp_opaque():
    """Treat the painted metal lamp as solid instead of applying its opacity atlas."""
    material = bpy.data.materials.get("M_Lamps_CCTV_2048.001")
    if not material or not material.use_nodes:
        raise RuntimeError("Expected the desk lamp's M_Lamps_CCTV_2048.001 material")

    principled = next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
    if not principled:
        raise RuntimeError("Desk lamp material has no Principled BSDF")

    alpha = principled.inputs.get("Alpha")
    if alpha:
        for link in list(alpha.links):
            material.node_tree.links.remove(link)
        alpha.default_value = 1.0

    # The asset's opacity atlas made the metal shade and support appear see-through.
    # Keep its color, roughness, metallic, and normal textures intact.
    material.surface_render_method = "DITHERED"


def optimize_export_textures(max_dimension: int = 2048):
    """Keep GLB texture memory reasonable for browsers and integrated GPUs."""
    resized = []
    for image in bpy.data.images:
        if image.source != "FILE":
            continue
        width, height = image.size
        longest_side = max(width, height)
        if longest_side <= max_dimension:
            continue
        if not image.has_data:
            image.reload()
        if not image.has_data:
            print("Skipped unloaded texture:", image.name)
            continue
        scale = max_dimension / longest_side
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        image.scale(new_width, new_height)
        image.pack()
        resized.append(f"{image.name}: {width}x{height} -> {new_width}x{new_height}")
    print("Resized export textures:", len(resized))
    for item in resized:
        print(" ", item)


def export_glb(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        export_materials="EXPORT",
        export_cameras=False,
        export_lights=False,
    )


def set_glb_alpha_modes(glb_path: Path):
    data = glb_path.read_bytes()
    if data[:4] != b"glTF":
        raise RuntimeError(f"Not a binary glTF: {glb_path}")

    version, total_length = struct.unpack_from("<II", data, 4)
    if version != 2 or total_length != len(data):
        raise RuntimeError("Unexpected GLB header")

    chunks, offset = [], 12
    document = None
    while offset < len(data):
        length, chunk_type = struct.unpack_from("<II", data, offset)
        payload = data[offset + 8:offset + 8 + length]
        chunks.append((chunk_type, payload))
        if chunk_type == 0x4E4F534A:
            document = json.loads(payload.decode("utf-8"))
        offset += 8 + length
    if document is None:
        raise RuntimeError("GLB JSON chunk not found")

    cutout_names = {"M_OfficeStool_Bin_2048", "M_Office_PinBoard_Photo_Notepad_1024"}
    opaque_names = {"M_Lamps_CCTV_2048.001"}
    changed = []
    made_opaque = []
    for material in document.get("materials", []):
        name = material.get("name")
        if name in opaque_names:
            material["alphaMode"] = "OPAQUE"
            material.pop("alphaCutoff", None)
            made_opaque.append(name)
        elif name in cutout_names and material.get("alphaMode") == "BLEND":
            material["alphaMode"] = "MASK"
            material["alphaCutoff"] = 0.5
            changed.append(name)

    if cutout_names - set(changed):
        missing = cutout_names - set(changed)
        if missing:
            raise RuntimeError(f"Cutout material(s) not converted from BLEND: {sorted(missing)}")
    if opaque_names - set(made_opaque):
        raise RuntimeError(f"Opaque material(s) missing from GLB: {sorted(opaque_names - set(made_opaque))}")

    json_chunk_index = next(i for i, (kind, _) in enumerate(chunks) if kind == 0x4E4F534A)
    encoded = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    chunks[json_chunk_index] = (0x4E4F534A, encoded)

    output = bytearray(b"glTF" + struct.pack("<II", 2, 0))
    for chunk_type, payload in chunks:
        output.extend(struct.pack("<II", len(payload), chunk_type))
        output.extend(payload)
    struct.pack_into("<I", output, 8, len(output))
    glb_path.write_bytes(output)
    print("Alpha MASK materials:", ", ".join(changed))
    print("Alpha OPAQUE materials:", ", ".join(made_opaque))


def main():
    args = parse_args()
    blend_path = Path(args.blend_output).resolve()
    glb_path = Path(args.glb_output).resolve()
    compose_front_view()
    fix_monitor_shell_shading()
    improve_table_shading()
    remove_legacy_table_grain()
    apply_real_tabletop_material()
    make_desk_lamp_opaque()
    add_coffee_mug()
    add_portfolio_lighting_rig()
    optimize_export_textures()
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    export_glb(glb_path)
    set_glb_alpha_modes(glb_path)
    print("Saved edited Blender scene:", blend_path)
    print("Exported portfolio model:", glb_path)


if __name__ == "__main__":
    main()
