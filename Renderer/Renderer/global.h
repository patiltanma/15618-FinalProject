#pragma once
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
//#include <math.h>


// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h


// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>
#include "erosion_kernel.h"

////////////////////////////////////////////////////////////////////////////////
//	OpenGL Constants
////////////////////////////////////////////////////////////////////////////////
// window display size constants
#define window_width 800
#define window_height 600

// animation constants
#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

// mesh display constants
#define mesh_width 1024
#define mesh_height 1024