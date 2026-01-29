from setuptools import setup, Extension
import os

# Define the extension module
module = Extension(
    'avantime_ray_tracer._extensions',
    sources=['src/extension.cpp'],
    extra_compile_args=['-std=c++11'],  # Ensure C++11 compliance
    language='c++'
)

setup(
    name='avantime_ray_tracer',
    version='0.1.0',
    description='Avantime Ray Tracer ported to Ironwood',
    author='Jules',
    packages=['avantime_ray_tracer'],
    ext_modules=[module],
    install_requires=[
        'jax',
        'jaxlib',
        'numpy',
    ],
)
