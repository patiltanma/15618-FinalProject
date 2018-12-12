#include "open_gl_utils.h"

static StopWatchInterface *timer = NULL;

// vbo variables
GLuint vbo_map = 0;
struct cudaGraphicsResource *cuda_vbo_resource_map;

GLuint vbo_new_map = 0;
struct cudaGraphicsResource *cuda_vbo_resource_new_map;

GLuint vbo_rain_map = 0;
struct cudaGraphicsResource *cuda_vbo_resource_rain_map;

//void *d_vbo_buffer = NULL;
float g_fAnim = 0.0;

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);
void cleanup();

bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags, cellData *map);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags, float *rain_map);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

////////////////////////////////////////////////////////////////////////////////
//! initialise and run openGL
////////////////////////////////////////////////////////////////////////////////
bool initAndRunGL(int argc, char **argv, cellData *map, cellData *new_map, float *rain_map)
{
	// Create the CUTIL timer
	sdkCreateTimer(&timer);

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	if (false == initGL(&argc, argv))
	{
		return false;
	}

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	{
		if (gpuGLDeviceInit(argc, (const char **)argv) == -1)
		{
			return false;
		}
	}
	else
	{
		cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
	}

	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutCloseFunc(cleanup);

	

	// create VBO
	createVBO(&vbo_map, &cuda_vbo_resource_map, cudaGraphicsMapFlagsWriteDiscard, map); // will read/write to this resource
	createVBO(&vbo_new_map, &cuda_vbo_resource_new_map, cudaGraphicsMapFlagsWriteDiscard, new_map); // will read/write to this resource
	createVBO(&vbo_rain_map, &cuda_vbo_resource_rain_map, cudaGraphicsMapFlagsWriteDiscard, rain_map); // will read/write to this resource

	// run the cuda part
	//runCuda(&cuda_vbo_resource);
	//runCuda(&cuda_vbo_resource, mesh_width, mesh_height, g_fAnim);
	//erodeCuda((float*)heightMap, 32);
	
	//runCuda(&cuda_vbo_resource, mesh_width, mesh_height, g_fAnim);
	erodeCuda(&cuda_vbo_resource_map, &cuda_vbo_resource_new_map, &cuda_vbo_resource_rain_map, 
		mesh_width, mesh_height);

	// start rendering mainloop
	glutMainLoop();


	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cuda GL Interop (VBO)");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	// initialize necessary OpenGL extensions
	if (!isGLVersionSupported(2, 0))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);

	SDK_CHECK_ERROR_GL();

	return true;
}



////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags, cellData *map)
{
	assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = mesh_width * mesh_height * sizeof(cellData);
	glBufferData(GL_ARRAY_BUFFER, size, map, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	SDK_CHECK_ERROR_GL();
}

void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags, float *rain_map)
{
	assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = mesh_width * mesh_height * sizeof(cellData);
	glBufferData(GL_ARRAY_BUFFER, size, rain_map, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

// Auto-Verification Code
static int fpsCount = 0;        // FPS count for averaging
static int fpsLimit = 1;        // FPS limit for sampling
static float avgFPS = 0.0f;
static unsigned int frameCount = 0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

void computeFPS(void);

extern char *sSDKsample;

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	sdkStartTimer(&timer);

	// run CUDA kernel to generate vertex positions
	//runCuda(&cuda_vbo_resource, mesh_width, mesh_height, g_fAnim);
	erodeCuda(&cuda_vbo_resource_map, &cuda_vbo_resource_new_map, &cuda_vbo_resource_rain_map,
		mesh_width, mesh_height);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo_map);
	//glVertexPointer(4, GL_FLOAT, 0, 0);
	glVertexPointer(3, GL_FLOAT, 0, 0);				//(number of elements for the vertex in the array,
													// type of the varibales in the array,
													// bytes of data between two vertex values of concern,
													// starting point)


	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);

	//glDrawArrays(GL_LINE_STRIP_ADJACENCY, 0, mesh_width * mesh_height);
	//glDrawArrays(GL_LINE_STRIP, 0, mesh_width * mesh_height);
	glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);

	/*
	struct vertex_per {
	float x;
	float y;
	float z;
	float ran;
	};

	//glBindBuffer(GL_ARRAY_BUFFER, vbo);
	vertex_per *data = (vertex_per *)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	for (int yy = 0; yy < mesh_height - 1; yy++) {
	//Makes OpenGL draw a triangle at every three consecutive vertices
	glBegin(GL_TRIANGLE_STRIP);
	for (int xx = 0; xx < mesh_width; xx++) {

	glVertex3f(data[xx].x, data[xx].y, data[xx].z);
	glVertex3f(data[xx+yy*mesh_width].x, data[xx + yy * mesh_width].y, data[xx + yy * mesh_width].z);

	/*
	Vec3f normal = _terrain->getNormal(x, );
	glNormal3f(normal[0], normal[1], normal[2]);
	glVertex3f(x, vbo[x + y * mesh_width], y);
	normal = _terrain->getNormal(x, z + 1);
	glNormal3f(normal[0], normal[1], normal[2]);
	glVertex3f(x, _terrain->getHeight(x, z + 1), z + 1);

	}
	glEnd();
	}
	*/
	//glDrawArrays(GL_TRIANGLES, 0, mesh_width * mesh_height);	//(what kind of primitve you want to draw,
	// starting point of the vertex in the array,
	// number of vertices you want to draw);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();

	g_fAnim += 0.01f;

	sdkStopTimer(&timer);
	computeFPS();
}





////////////////////////////////////////////////////////////////////////////////
//! glut exit call back function; called when glut window exits 
////////////////////////////////////////////////////////////////////////////////
void cleanup()
{
	sdkDeleteTimer(&timer);

	if (vbo_map)
	{
		deleteVBO(&vbo_map, cuda_vbo_resource_map);
	}

	if (vbo_new_map)
	{
		deleteVBO(&vbo_new_map, cuda_vbo_resource_new_map);
	}

	if (vbo_rain_map)
	{
		deleteVBO(&vbo_rain_map, cuda_vbo_resource_rain_map);
	}
}


////////////////////////////////////////////////////////////////////////////////
//! Timer events handler
////////////////////////////////////////////////////////////////////////////////
void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();								//the window's display callback will be called to redisplay the window's normal plane
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case (27):
#if defined(__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
		return;
#endif
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}


////////////////////////////////////////////////////////////////////////////////
//! motion event handlers called when mouse moves with one or more buttons pressed
////////////////////////////////////////////////////////////////////////////////
void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}


////////////////////////////////////////////////////////////////////////////////
//! Computes FPS and sets the window name with the current FPS
////////////////////////////////////////////////////////////////////////////////
void computeFPS(void)
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		fpsCount = 0;
		fpsLimit = (int)MAX(avgFPS, 1.f);

		sdkResetTimer(&timer);
	}

	char fps[256];
	sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
	glutSetWindowTitle(fps);
}

