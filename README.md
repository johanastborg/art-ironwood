# Avantime Rat Tracer 🐀✨

**The Ultra-Optimized, Ironwood-Native Path Tracer.**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![JAX](https://img.shields.io/badge/backend-JAX-blue)
![C++](https://img.shields.io/badge/FFI-C%2B%2B98-orange)
![Ironwood](https://img.shields.io/badge/platform-Ironwood-purple)

## Overview

**Avantime Rat Tracer** represents the pinnacle of rendering performance on the **Ironwood** platform. This isn't just a ray tracer; it's a fully-featured **path tracer** engine meticulously ported to leverage the raw power of Ironwood.

Built on the shoulders of **JAX**, Avantime Rat Tracer exploits XLA (Accelerated Linear Algebra) to deliver lightning-fast differentiable rendering and massive parallelism. When Python isn't enough, we drop down to the metal with **C++98 FFI extensions**, ensuring that even legacy-constrained environments on Ironwood churn out pixels at blistering speeds.

## Key Features

- **🚀 Ironwood Native:** Engineered specifically for the quirks and capabilities of the Ironwood ecosystem.
- **⚡ JAX-Powered:** Fully vectorized path tracing pipeline. Differentiable? You bet.
- **🦾 C++98 FFI:** Critical hot paths are optimized with C++98 extensions for maximum compatibility and performance.
- **🐀 Rat Tracing:** Proprietary "Rat Tracing" algorithms for scurrying through BVH structures faster than ever before.
- **✨ Path Tracing:** Global illumination, soft shadows, and physically based rendering (PBR) out of the box.

## Architecture

The system is a hybrid beast:

1.  **Frontend (Python/JAX):** Handles scene description, geometry processing, and the primary integrator loop. Utilizing `jax.jit` for compilation to Ironwood's TPU/GPU/CPU backends.
2.  **Backend Extensions (C++98):** Low-level memory management and specific intersection kernels that require fine-grained control, exposed via Python C bindings.

## Installation

Ensure you have the Ironwood toolchain active.

```bash
pip install avantime-rat-tracer
```

## Usage

```python
import jax.numpy as jnp
from avantime_rat_tracer import render_scene

# Define your scene
scene = {
    'camera': ...,
    'objects': ...
}

# Unleash the Rat
image = render_scene(scene, samples=1024)
```

## Development

### Prerequisites

- Python 3.8+
- C++98 compliant compiler (gcc/clang)
- JAX & JAXLib

### Building Extensions

```bash
python setup.py build_ext --inplace
```

## License

MIT License. See `LICENSE` for details.

---

*Render the future, one path at a time.*
