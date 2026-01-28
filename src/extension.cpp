#include <Python.h>
#include <cmath>

// A simple C++98 compliant function to demonstrate FFI
// This could be a complex ray-box intersection or noise function in the future

static PyObject* fast_inverse_sqrt(PyObject* self, PyObject* args) {
    float number;
    if (!PyArg_ParseTuple(args, "f", &number)) {
        return NULL;
    }

    // Classic fast inverse square root approximation (Quake III style, adapted)
    int i;
    float x2, y;
    const float threehalfs = 1.5F;

    x2 = number * 0.5F;
    y  = number;
    i  = * ( int * ) &y;                        // evil floating point bit level hacking
    i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
    y  = * ( float * ) &i;
    y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration

    return Py_BuildValue("f", y);
}

// Method definition object for this extension, these methods are the publicly accessible methods of your extension
static PyMethodDef ExtensionMethods[] = {
    {"fast_inverse_sqrt", fast_inverse_sqrt, METH_VARARGS, "Calculate fast inverse square root (C++98 style)"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef extensionmodule = {
    PyModuleDef_HEAD_INIT,
    "_extensions",
    "Internal C++ extensions for Avantime Ray Tracer",
    -1,
    ExtensionMethods
};

// Module initialization
PyMODINIT_FUNC PyInit__extensions(void) {
    return PyModule_Create(&extensionmodule);
}
