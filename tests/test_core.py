import pytest
from avantime_ray_tracer.core import render_scene
import jax.numpy as jnp

def test_render_scene_structure():
    image = render_scene(None, samples=1)
    assert image.shape == (600, 800, 3)
    assert isinstance(image, jnp.ndarray)
