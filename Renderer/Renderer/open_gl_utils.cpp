#include "open_gl_utils.h"
#include "vec3f.h"

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

Vec3f computeNormals(int x, int y, cellData* h);

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
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
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
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_NORMALIZE);

	// enabling glColor as it doenot work with GL_LIGHTING
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

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
	glBufferData(GL_ARRAY_BUFFER, size, map, GL_DYNAMIC_COPY);

	//glBindBuffer(GL_ARRAY_BUFFER, 0);

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
	unsigned int size = mesh_width * mesh_height * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, rain_map, GL_DYNAMIC_COPY);
	
	//glBindBuffer(GL_ARRAY_BUFFER, 0);

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
void drawPoints(cellData *data);
void drawTriangles(cellData *data);
void drawPointsHeightColored(cellData *data);

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
	//glScalef();

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo_map);
	
	/*
	//glVertexPointer(4, GL_FLOAT, 0, 0);
	glVertexPointer(3, GL_FLOAT, 0, 0);				//(number of elements for the vertex in the array,
													// type of the varibales in the array,
													// bytes of data between two vertex values of concern,
													// starting point)
	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
	*/

	cellData *data = (cellData *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);

	//drawPoints(data);
	drawPointsHeightColored(data);
	//drawTriangles(data);
	//glColor3f(1.0, 1.0, 0.0);
	
	glUnmapBuffer(GL_ARRAY_BUFFER);

	glutSwapBuffers();

	g_fAnim += 0.01f;

	sdkStopTimer(&timer);
	computeFPS();
}

void drawPoints(cellData *data)
{
	for (int yy = 0; yy < mesh_height - 1; yy++) {
		//Makes OpenGL draw a triangle at every three consecutive vertices
		glBegin(GL_POINTS);
		for (int xx = 0; xx < mesh_width - 1; xx++) {

			float u = xx / (float)mesh_width;
			float v = yy / (float)mesh_height;
			u = u * 2.0f - 1.0f;
			v = v * 2.0f - 1.0f;


			if (data[xx + yy * mesh_width].water_vol != 0.0f)
			{
				//printf("%f\n", data[xx + yy * mesh_width].water_vol);
				glColor3f(0.0, 0.0, 1.0);
				glVertex3f(u, v, data[xx + yy * mesh_width].water_height / 100);
			}

			glColor3f(1.0, 1.0, 0.0);
			glVertex3f(u, v, (data[xx + yy * mesh_width].height) / 100);

			//printf("%f\n", data[xx + yy * mesh_width].height);
		}
		glEnd();
	}
}

void drawPointsHeightColored(cellData *data)
{
	for (int yy = 0; yy < mesh_height - 1; yy++) {
		//Makes OpenGL draw a triangle at every three consecutive vertices
		glBegin(GL_POINTS);
		for (int xx = 0; xx < mesh_width - 1; xx++) {

			float u = xx / (float)mesh_width;
			float v = yy / (float)mesh_height;
			u = u * 2.0f - 1.0f;
			v = v * 2.0f - 1.0f;


			if (data[xx + yy * mesh_width].water_vol != 0.0f)
			{
				//printf("%f\n", data[xx + yy * mesh_width].water_vol);
				glColor3f(0.0, 0.0, 1.0);
				glVertex3f(u, v, data[xx + yy * mesh_width].water_height / 200);
			}
			float h = data[xx + yy * mesh_width].height/100;
			//float r = 1, g = 1, b = 1;
			/*
			if (h < 33)
			{
				r = 100 / h;
			}
			else if (h < 66)
			{
				g = 100 / h;
			}
			else
			{
				b = 100 / h;
			}
			*/
			glColor3f(h, 0.5, 0.5);
			glVertex3f(u, v, h);

			//printf("%f\n", data[xx + yy * mesh_width].height);
		}
		glEnd();
	}
}

void drawTriangles(cellData *data)
{
	glColor3f(0.3f, 0.9f, 0.0f);
	for (int yy = 0; yy < mesh_height - 1; yy++) {
		//Makes OpenGL draw a triangle at every three consecutive vertices
		glBegin(GL_TRIANGLE_STRIP);
		for (int xx = 0; xx < mesh_width; xx++) {
			float u = xx / (float)mesh_width;
			float v = yy / (float)mesh_height;
			u = u * 2.0f - 1.0f;
			v = v * 2.0f - 1.0f;

			Vec3f normal = computeNormals(xx, yy, data);
			glNormal3f(normal[0], normal[1], normal[2]);
			glVertex3f(u, v, data[xx + yy * mesh_width].height / 100);

			v = (yy + 1) / (float)mesh_height;
			v = v * 2.0f - 1.0f;
			normal = computeNormals(xx, yy + 1, data);
			glNormal3f(normal[0], normal[1], normal[2]);
			glVertex3f(u, v, data[xx + (yy + 1) * mesh_width].height / 100);
		}
		glEnd();
	}
}


Vec3f computeNormals(int x, int y, cellData* h)
{
	//Vec3f normals2;
	Vec3f sum(0.0f, 0.0f, 0.0f);

	Vec3f out;
	if (y > 0) {
		//out = Vec3f(0.0f, hs[z - 1][x] - hs[z][x], -1.0f);
		out = Vec3f(0.0f, h[x + (y-1) * mesh_width].water_height - h[x + y * mesh_width].water_height, -1.0f);
	}

	Vec3f in;
	if (y < mesh_height - 1) {
		//in = Vec3f(0.0f, hs[z + 1][x] - hs[z][x], 1.0f);
		in = Vec3f(0.0f, h[x + (y + 1) * mesh_width].water_height - h[x + y * mesh_width].water_height, 1.0f);
	}

	Vec3f left;
	if (x > 0) {
		//left = Vec3f(-1.0f, hs[z][x - 1] - hs[z][x], 0.0f);
		left = Vec3f(-1.0f, h[(x - 1) + y * mesh_width].water_height - h[x + y * mesh_width].water_height, 0.0f);
	}

	Vec3f right;
	if (x < mesh_width - 1) {
		//right = Vec3f(1.0f, hs[z][x + 1] - hs[z][x], 0.0f);
		right = Vec3f(1.0f, h[(x + 1) + y * mesh_width].water_height - h[x + y * mesh_width].water_height, 0.0f);
	}

	if (x > 0 && y > 0) {
		sum += out.cross(left).normalize();
	}
	if (x > 0 && y < mesh_height - 1) {
		sum += left.cross(in).normalize();
	}
	if (x < mesh_width - 1 && y < mesh_height - 1) {
		sum += in.cross(right).normalize();
	}
	if (x < mesh_width - 1 && y > 0) {
		sum += right.cross(out).normalize();
	}

	return sum;
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

