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

def refract(uv, n, etai_over_etat):
    cos_theta = jnp.minimum(jnp.dot(-uv, n), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -jnp.sqrt(jnp.abs(1.0 - jnp.dot(r_out_perp, r_out_perp))) * n
    return r_out_perp + r_out_parallel

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
            roughness = s[8]
            # transmission: index 9, ior: index 10 (defaults if not present)
            # We assume scene array is sized 11 now
            transmission = s[9]
            ior = s[10]
            return normal, albedo, reflectivity, roughness, transmission, ior

        def get_plane_data(idx):
            normal = plane[3:6]
            albedo = get_checkerboard_color(hit_point)
            reflectivity = 0.3
            roughness = 0.5
            transmission = 0.0
            ior = 1.0
            return normal, albedo, reflectivity, roughness, transmission, ior

        normal, albedo, reflectivity, roughness, transmission, ior = jax.lax.cond(
            hit_obj == 1,
            lambda: get_sphere_data(closest_sphere_idx),
            lambda: get_plane_data(0)
        )

        # Calculate Lighting (Local)
        to_light = normalize(light_pos - hit_point)
        shadow_sphere_dists = jax.vmap(lambda s: sphere_intersect(s[:3], s[3], hit_point, to_light))(spheres)
        in_shadow = jnp.min(shadow_sphere_dists) < jnp.linalg.norm(light_pos - hit_point)

        # BRDF Evaluation (for non-transmitted part)
        view_dir = -ray_dir
        brdf_color = eval_brdf(normal, view_dir, to_light, albedo, roughness)
        
        # Apply light color/intensity
        light_color = scene['light']['color']
        brdf_color = brdf_color * light_color

        local_color = jax.lax.select(in_shadow, jnp.array([0.0, 0.0, 0.0]), brdf_color)

        # Reflection
        reflect_dir = normalize(ray_dir - 2.0 * jnp.dot(ray_dir, normal) * normal)
        reflect_origin = hit_point + reflect_dir * 0.001
        reflected_color = trace(reflect_origin, reflect_dir, scene, depth - 1, key)

        # Refraction
        def do_refraction():
            refraction_ratio = jnp.where(jnp.dot(ray_dir, normal) > 0, ior, 1.0 / ior)
            corrected_normal = jnp.where(jnp.dot(ray_dir, normal) > 0, -normal, normal)

            cos_theta = jnp.minimum(jnp.dot(-ray_dir, corrected_normal), 1.0)
            sin_theta = jnp.sqrt(1.0 - cos_theta**2)

            cannot_refract = refraction_ratio * sin_theta > 1.0

            # Schlick approximation for reflectivity vs transmission mix
            r0 = (1.0 - refraction_ratio) / (1.0 + refraction_ratio)
            r0 = r0**2
            schlick_reflectivity = r0 + (1.0 - r0) * (1.0 - cos_theta)**5

            # If TIR or simple reflection dominates
            # But we already have reflectivity parameter.
            # Usually for glass, reflectivity is Fresnel.
            # Here we mix: local_color (diffuse) + reflection + refraction

            refract_dir = refract(ray_dir, corrected_normal, refraction_ratio)
            refract_origin = hit_point + refract_dir * 0.001
            refracted_color_val = trace(refract_origin, refract_dir, scene, depth - 1, key)

            # If total internal reflection, result is reflection
            final_refract = jax.lax.select(cannot_refract, reflected_color, refracted_color_val)

            # Composite
            # 1. Diffuse component is suppressed by transmission
            # 2. Specular (reflection) is added based on Fresnel or reflectivity param

            # Simplified composition:
            # (Diffuse) * (1 - trans - refl) + (Reflection) * (refl + fresnel) + (Refraction) * trans

            return final_refract

        # Only compute refraction if transmission > 0
        refracted_color = jax.lax.cond(
            transmission > 0.0,
            do_refraction,
            lambda: jnp.array([0.0, 0.0, 0.0])
        )

        # Final mix
        # If trans > 0, we assume it's a dielectric/glass-like material.
        # Albedo acts as tint for refraction, diffuse is small or zero usually.
        # But `local_color` contains diffuse + specular highlight.
        # For glass, we want specular highlight + refraction + tint.

        dielectric_color = refracted_color * albedo # Tint

        # If transmission > 0, mix refraction and reflection. Ignore diffuse (local_color) except for highlight?
        # Let's keep it simple:
        # result = local_color * (1 - trans - refl) + reflection * refl + refraction * trans

        mixed_color = local_color * (1.0 - transmission - reflectivity) + \
                      reflected_color * reflectivity + \
                      refracted_color * transmission

        return mixed_color

    return jax.lax.cond(hit_dist == jnp.inf, lambda: jnp.array([0.0, 0.0, 0.0]), perform_shading)

def render_scene(scene_dict=None, samples=1, camera_origin=None, camera_target=None):
    """
    Renders the scene. If scene_dict is None, uses the default "basic scene".
    """
    if scene_dict is None:
        # Default scene definition
        # Spheres: x, y, z, r, r_col, g_col, b_col, reflectivity, roughness, transmission, ior
        spheres = jnp.array([
            [-1.2, 0.5, -3.0, 0.5, 1.0, 0.0, 0.0, 0.5, 0.1, 0.0, 1.0], # Red, shiny
            [ 0.0, 0.5, -3.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.4, 0.0, 1.0], # Green, rougher
            [ 1.2, 0.5, -3.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.05, 0.0, 1.0], # Blue, very shiny
            # New Magenta Glass Sphere
            [ 0.0, 1.0, -1.5, 1.0, 1.0, 0.0, 1.0, 0.1, 0.05, 0.75, 1.5], # Magenta, Large, Transparent
        ])
        # Plane: x, y, z (point), nx, ny, nz (normal)
        plane = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        scene_dict = {
            'spheres': spheres,
            'plane': plane,
            'light': {
                'pos': jnp.array([5.0, 10.0, -5.0]), # Higher intensity/position? Color is still 1,1,1
                'color': jnp.array([1.5, 1.5, 1.5]) # Increased intensity (1.5)
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
    if camera_origin is None:
        origin = jnp.array([0.0, 0.8, -0.5])
    else:
        origin = jnp.array(camera_origin)

    if camera_target is None:
        target = jnp.array([0.0, 0.5, -3.0])
    else:
        target = jnp.array(camera_target)

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
