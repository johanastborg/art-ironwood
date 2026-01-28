import jax
import jax.numpy as jnp
from functools import partial

# Constants
WIDTH = 800
HEIGHT = 600
MAX_DEPTH = 3

def normalize(v):
    return v / jnp.linalg.norm(v)

def sphere_intersect(center, radius, ray_origin, ray_dir):
    oc = ray_origin - center
    a = jnp.dot(ray_dir, ray_dir)
    b = 2.0 * jnp.dot(oc, ray_dir)
    c = jnp.dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c

    # Return infinity if no intersection
    dist = jnp.where(discriminant < 0, jnp.inf, (-b - jnp.sqrt(discriminant)) / (2.0 * a))
    dist = jnp.where(dist > 0.001, dist, jnp.inf) # Avoid self-intersection
    return dist

def plane_intersect(point, normal, ray_origin, ray_dir):
    denom = jnp.dot(normal, ray_dir)
    # Avoid division by zero and back-facing planes if needed,
    # but for ground plane visible from top, denom should be negative
    dist = jnp.dot(point - ray_origin, normal) / (denom + 1e-6)
    dist = jnp.where((dist > 0.001) & (jnp.abs(denom) > 1e-6), dist, jnp.inf)
    return dist

def get_checkerboard_color(hit_point):
    # Scale for checkerboard
    scale = 2.0
    u = jnp.floor(hit_point[0] * scale)
    v = jnp.floor(hit_point[2] * scale) # Z is depth
    is_white = (u + v) % 2 == 0
    return jnp.where(is_white, jnp.array([1.0, 1.0, 1.0]), jnp.array([0.0, 0.0, 0.0]))

def eval_brdf(normal, view_dir, light_dir, albedo, roughness):
    """
    Evaluates a simple BRDF (Diffuse + Blinn-Phong Specular).
    """
    # Diffuse (Lambertian)
    NdotL = jnp.maximum(0.0, jnp.dot(normal, light_dir))
    diffuse = albedo * NdotL

    # Specular (Blinn-Phong)
    half_vec = normalize(light_dir + view_dir)
    NdotH = jnp.maximum(0.0, jnp.dot(normal, half_vec))

    # Map roughness (0-1) to shininess (1-100+)
    # Roughness 0 -> Shininess 500 (Sharp)
    # Roughness 1 -> Shininess 1 (Dull)
    shininess = 500.0 * (1.0 - roughness) + 1.0

    spec_intensity = jnp.power(NdotH, shininess)

    # Simple Fresnel approximation (Schlick) could be added here,
    # but strictly Blinn-Phong is just intensity.
    specular = jnp.array([1.0, 1.0, 1.0]) * spec_intensity * (1.0 - roughness) # Scale by smoothness

    return diffuse + specular

def trace(ray_origin, ray_dir, scene, depth, key):
    if depth == 0:
        return jnp.array([0.0, 0.0, 0.0])

    spheres = scene['spheres']
    plane = scene['plane']
    light_pos = scene['light']['pos']

    # Intersections
    sphere_dists = jax.vmap(lambda s: sphere_intersect(s[:3], s[3], ray_origin, ray_dir))(spheres)
    closest_sphere_idx = jnp.argmin(sphere_dists)
    min_sphere_dist = sphere_dists[closest_sphere_idx]

    plane_dist = plane_intersect(plane[:3], plane[3:6], ray_origin, ray_dir)

    hit_obj = jnp.where(min_sphere_dist < plane_dist, 1, 2) # 1=Sphere, 2=Plane
    hit_dist = jnp.minimum(min_sphere_dist, plane_dist)

    # If no hit, return background immediately (black)
    def perform_shading():
        # Hit Point
        hit_point = ray_origin + hit_dist * ray_dir

        # Normal and Material
        def get_sphere_data(idx):
            s = spheres[idx]
            normal = normalize(hit_point - s[:3])
            albedo = s[4:7]
            reflectivity = s[7]
            # Assumes spheres now have roughness at index 8. If not present (old scene data), default to 0.1
            # But JAX array must be fixed size. We will update scene generation.
            roughness = s[8]
            return normal, albedo, reflectivity, roughness

        def get_plane_data(idx):
            normal = plane[3:6]
            albedo = get_checkerboard_color(hit_point)
            reflectivity = 0.3
            roughness = 0.5 # Floor is somewhat rough
            return normal, albedo, reflectivity, roughness

        normal, albedo, reflectivity, roughness = jax.lax.cond(
            hit_obj == 1,
            lambda: get_sphere_data(closest_sphere_idx),
            lambda: get_plane_data(0)
        )

        # Calculate Lighting (Local)
        to_light = normalize(light_pos - hit_point)
        shadow_sphere_dists = jax.vmap(lambda s: sphere_intersect(s[:3], s[3], hit_point, to_light))(spheres)
        in_shadow = jnp.min(shadow_sphere_dists) < jnp.linalg.norm(light_pos - hit_point)

        # BRDF Evaluation
        view_dir = -ray_dir
        brdf_color = eval_brdf(normal, view_dir, to_light, albedo, roughness)

        local_color = jax.lax.select(in_shadow, jnp.array([0.0, 0.0, 0.0]), brdf_color)

        # Reflection
        reflect_dir = normalize(ray_dir - 2.0 * jnp.dot(ray_dir, normal) * normal)
        # Recursion
        # Note: We need to nudge the origin to avoid self-intersection
        reflect_origin = hit_point + reflect_dir * 0.001
        reflected_color = trace(reflect_origin, reflect_dir, scene, depth - 1, key)

        return local_color * (1.0 - reflectivity) + reflected_color * reflectivity

    return jax.lax.cond(hit_dist == jnp.inf, lambda: jnp.array([0.0, 0.0, 0.0]), perform_shading)

def render_scene(scene_dict=None, samples=1):
    """
    Renders the scene. If scene_dict is None, uses the default "basic scene".
    """
    if scene_dict is None:
        # Default scene definition
        # Spheres: x, y, z, r, r_col, g_col, b_col, reflectivity, roughness
        spheres = jnp.array([
            [-1.2, 0.5, -3.0, 0.5, 1.0, 0.0, 0.0, 0.5, 0.1], # Red, shiny
            [ 0.0, 0.5, -3.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.4], # Green, rougher
            [ 1.2, 0.5, -3.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.05], # Blue, very shiny
        ])
        # Plane: x, y, z (point), nx, ny, nz (normal)
        # Note: Plane data structure remains same for now as we hardcode roughness in get_plane_data
        plane = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        scene_dict = {
            'spheres': spheres,
            'plane': plane,
            'light': {
                'pos': jnp.array([5.0, 5.0, -5.0]), # Light position
                'color': jnp.array([1.0, 1.0, 1.0])
            }
        }

    print(f"Initializing Ray Tracer with {samples} samples...")

    # Camera setup
    aspect_ratio = WIDTH / HEIGHT

    # Generate rays
    y, x = jnp.mgrid[0:HEIGHT, 0:WIDTH]
    # Normalize pixel coordinates
    uv_x = (x / WIDTH) * 2.0 - 1.0
    uv_y = ((HEIGHT - y) / HEIGHT) * 2.0 - 1.0
    uv_x *= aspect_ratio

    # Camera pos and dir
    origin = jnp.array([0.0, 1.5, 1.0])
    target = jnp.array([0.0, 0.5, -3.0])

    cam_forward = normalize(target - origin)
    cam_right = normalize(jnp.cross(cam_forward, jnp.array([0.0, 1.0, 0.0])))
    cam_up = jnp.cross(cam_right, cam_forward)

    ray_dirs = normalize(
        uv_x[..., None] * cam_right +
        uv_y[..., None] * cam_up +
        cam_forward
    )

    ray_origins = jnp.broadcast_to(origin, ray_dirs.shape)

    # Render function (vectorized)
    flat_origins = ray_origins.reshape(-1, 3)
    flat_dirs = ray_dirs.reshape(-1, 3)

    key = jax.random.PRNGKey(0)

    render_func = jax.jit(lambda o, d: trace(o, d, scene_dict, MAX_DEPTH, key))

    print("Tracing rays...")
    pixel_colors = jax.vmap(render_func)(flat_origins, flat_dirs)

    image = pixel_colors.reshape(HEIGHT, WIDTH, 3)
    image = jnp.clip(image, 0.0, 1.0)

    return image
