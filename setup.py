from setuptools import setup, Extension
import os

# Define the extension module
module = Extension(
    'avantime_rat_tracer._extensions',
    sources=['src/extension.cpp'],
    extra_compile_args=['-std=c++98'],  # Ensure C++98 compliance
    language='c++'
)

setup(
    name='avantime_rat_tracer',
    version='0.1.0',
    description='Avantime Rat Tracer ported to Ironwood',
    author='Jules',
    packages=['avantime_rat_tracer'],
    ext_modules=[module],
    install_requires=[
        'jax',
        'jaxlib',
        'numpy',
    ],
)
