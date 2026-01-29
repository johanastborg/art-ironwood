from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jax.numpy as jnp
from avantime_ray_tracer.core import render_scene
import io
from PIL import Image
from fastapi.responses import Response
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Light(BaseModel):
    pos: list[float]
    color: list[float]

class Sphere(BaseModel):
    center: list[float]
    radius: float
    color: list[float]
    reflectivity: float
    roughness: float
    transmission: float
    ior: float

class SceneRequest(BaseModel):
    spheres: list[Sphere]
    light_pos: list[float]
    light_intensity: float = 1.5
    camera_origin: list[float] | None = None
    camera_target: list[float] | None = None

@app.get("/")
def read_root():
    return {"message": "Avantime Ray Tracer API"}

@app.post("/render")
def render(request: SceneRequest):
    try:
        # Construct scene dictionary
        spheres_data = []
        for s in request.spheres:
            # x, y, z, r, r_col, g_col, b_col, refl, rough, trans, ior
            spheres_data.append([
                s.center[0], s.center[1], s.center[2],
                s.radius,
                s.color[0], s.color[1], s.color[2],
                s.reflectivity,
                s.roughness,
                s.transmission,
                s.ior
            ])

        spheres_array = jnp.array(spheres_data)

        # Hardcoded plane for now
        plane_array = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        light_pos = jnp.array(request.light_pos)
        # Use requested intensity
        val = request.light_intensity
        light_color = jnp.array([val, val, val])

        scene = {
            'spheres': spheres_array,
            'plane': plane_array,
            'light': {
                'pos': light_pos,
                'color': light_color
            }
        }

        image_data = render_scene(
            scene, 
            samples=1, 
            camera_origin=request.camera_origin,
            camera_target=request.camera_target
        )

        # Convert to PNG
        image_data = (image_data * 255).astype(jnp.uint8)
        # JAX array to numpy for PIL
        image_np =  image_data.__array__()

        img = Image.fromarray(image_np, 'RGB')

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return Response(content=img_byte_arr, media_type="image/png")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
