#include "open_gl_utils.h"
#include "vec3f.h"

static StopWatchInterface *timer = NULL;

// vbo variables
GLuint vbo_height = 0;
struct cudaGraphicsResource *cvr_height; //CVR = cuda vbo resource

GLuint vbo_water = 0;
struct cudaGraphicsResource *cvr_water;

GLuint vbo_sediment = 0;
struct cudaGraphicsResource *cvr_sediment;

GLuint vbo_new_height = 0;
struct cudaGraphicsResource *cvr_new_height; //CVR = cuda vbo resource

GLuint vbo_new_water = 0;
struct cudaGraphicsResource *cvr_new_water;

GLuint vbo_new_sediment = 0;
struct cudaGraphicsResource *cvr_new_sediment;

GLuint vbo_rain = 0;
struct cudaGraphicsResource *cvr_rain;

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

Vec3f computeNormals(int x, int y, float* h);
void print(float x, float y, char *string);

////////////////////////////////////////////////////////////////////////////////
//! initialise and run openGL
////////////////////////////////////////////////////////////////////////////////
bool initAndRunGL(int argc, char **argv, cellData *map, cellData *new_map, float *rain)
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

	// create VBOs
	createVBO(&vbo_height, &cvr_height, cudaGraphicsMapFlagsWriteDiscard, map->height);					// will read/write to this resource
	createVBO(&vbo_water, &cvr_water, cudaGraphicsMapFlagsWriteDiscard, map->water);					// will read/write to this resource
	createVBO(&vbo_sediment, &cvr_sediment, cudaGraphicsMapFlagsWriteDiscard, map->sediment);			// will read/write to this resource

	createVBO(&vbo_new_height, &cvr_new_height, cudaGraphicsMapFlagsWriteDiscard, new_map->height);					// will read/write to this resource
	createVBO(&vbo_new_water, &cvr_new_water, cudaGraphicsMapFlagsWriteDiscard, new_map->water);				// will read/write to this resource
	createVBO(&vbo_new_sediment, &cvr_new_sediment, cudaGraphicsMapFlagsWriteDiscard, new_map->sediment);		// will read/write to this resource

	createVBO(&vbo_rain, &cvr_rain, cudaGraphicsMapFlagsWriteDiscard, rain);


	erodeCuda(&cvr_height, &cvr_water, &cvr_sediment,
		&cvr_new_height, &cvr_new_water, &cvr_new_sediment,
		&cvr_rain, MESH_DIM);

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
	glOrtho(-3.0, 3.0, -3.0, 3.0, -3.0, 3.0);


	GLfloat position0[] = { 0.1, 0.1, 10.0, 0.0 };
	GLfloat diffuse0[] = { 0.8, 0.8, 0.8, 1.0 }; 
	GLfloat specular0[] = { 1.0, 1.0, 1.0, 1.0 }; // Is term - White
	GLfloat ambient0[] = { 0.2, 0.2, 0.2, 1.0 }; // Ia term - Gray
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glLightfv(GL_LIGHT0, GL_POSITION, position0);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse0);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specular0);
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambient0);
	glEnable(GL_NORMALIZE);

	//GLfloat light_pos[] = { 0.0, 1.0, 0.0, 0.0 };
	//glLightfv(GL_LIGHT0, GL_POSITION, light_pos);


	// enabling glColor as it doenot work with GL_LIGHTING
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

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
	unsigned int size = MESH_DIM * MESH_DIM * sizeof(cellData);
	glBufferData(GL_ARRAY_BUFFER, size, map, GL_DYNAMIC_COPY);

	//glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	SDK_CHECK_ERROR_GL();
}

void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags, float *map)
{
	assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = MESH_DIM * MESH_DIM * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, map, GL_DYNAMIC_COPY);
	
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
float rotate_x = 0.0, rotate_y = 0.0, rotate_z = 0.0;
float translate_z = -3.0;

void computeFPS(void);
void drawPoints(float *data, float *water);
void drawTriangles(float *data, float *water);
void drawPointsHeightColored(float *data);

extern char *sSDKsample;


// keyboard control varibles
static int iterations = 0;
static bool displayWaterFlag = true;
static bool pauseFlag = false;

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	if (pauseFlag) { return; }

	if (iterations > ITERATIONS)
	{
		print(0.1f, 0.1f,"Press 'p' to resume to further iteration.");
		return;
	}
	iterations++;

	sdkStartTimer(&timer);
	

	

	// run CUDA kernel to generate vertex positions
	//runCuda(&cuda_vbo_resource, mesh_width, mesh_height, g_fAnim);
	erodeCuda(&cvr_height, &cvr_water, &cvr_sediment,
		&cvr_new_height, &cvr_new_water, &cvr_new_sediment,
		&cvr_rain, MESH_DIM);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);
	glRotatef(rotate_z, 0.0, 0.0, 1.0);
	//glScalef();

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo_height);
	float *height = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_water);
	float *water = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

	//drawPoints(height, water);
	//drawPointsHeightColored(height);
	drawTriangles(height, water);
	
	// unmap vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo_height);
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_water);
	glUnmapBuffer(GL_ARRAY_BUFFER);

	glutSwapBuffers();


	
	g_fAnim += 0.01f;

	sdkStopTimer(&timer);
	computeFPS();
}

void drawPoints(float *data, float *water)
{
	for (int yy = 0; yy < MESH_DIM - 1; yy++) {
		//Makes OpenGL draw a triangle at every three consecutive vertices
		glBegin(GL_POINTS);
		for (int xx = 0; xx < MESH_DIM - 1; xx++) {

			float u = xx / (float)MESH_DIM;
			float v = yy / (float)MESH_DIM;
			u = u * 2.0f - 1.0f;
			v = v * 2.0f - 1.0f;

			
			if (water[xx + yy * MESH_DIM] != 0.0f)
			{
				//printf("%f\n", data[xx + yy * mesh_width].water_vol);
				glColor3f(0.0, 0.0, 1.0);
				glVertex3f(u, v, (data[xx + yy * MESH_DIM] + water[xx + yy * MESH_DIM]) / 100);
			}
			

			glColor3f(0.5f, 0.35f, 0.05f);
			glVertex3f(u, v, data[xx + yy * MESH_DIM] / 100);

			//printf("%f\n", data[xx + yy * mesh_width].height);
		}
		glEnd();
	}
}

void drawPointsHeightColored(float *data)
{
	for (int yy = 0; yy < MESH_DIM - 1; yy++) {
		//Makes OpenGL draw a triangle at every three consecutive vertices
		glBegin(GL_POINTS);
		for (int xx = 0; xx < MESH_DIM - 1; xx++) {

			float u = xx / (float)MESH_DIM;
			float v = yy / (float)MESH_DIM;
			u = u * 2.0f - 1.0f;
			v = v * 2.0f - 1.0f;

			/*
			if (data[xx + yy * MESH_DIM].water_vol != 0.0f)
			{
				//printf("%f\n", data[xx + yy * mesh_width].water_vol);
				glColor3f(0.0, 0.0, 1.0);
				glVertex3f(u, v, data[xx + yy * mesh_width].water_height / 200);
			}
			*/
			float h = data[xx + yy * MESH_DIM] / 100;
			glColor3f(h, 0.5, 0.5);
			glVertex3f(u, v, h);

			//printf("%f\n", data[xx + yy * mesh_width].height);
		}
		glEnd();
	}
}

void drawTriangles(float *data, float *water)
{
	/*
	glColor3f(0.0f, 0.0f, 0.1f);
	for (int yy = 0; yy < MESH_DIM - 1; yy++) {
		//Makes OpenGL draw a triangle at every three consecutive vertices
		glBegin(GL_TRIANGLE_STRIP);
		for (int xx = 0; xx < MESH_DIM; xx++) {
			float u = xx / (float)MESH_DIM;
			float v = yy / (float)MESH_DIM;
			
			u = u * 2.0f - 1.0f;
			v = v * 2.0f - 1.0f;
			
			float water_1 = water[xx + yy * MESH_DIM] / 100;
			float water_2 = water[xx + (yy + 1) * MESH_DIM] / 100;
			if ((water_1 * water_2) == 0.0f)
			{
				// fold the triangle on itself if no water
				glVertex3f(u, v, water_1 + data[xx + yy * MESH_DIM] / 100);
				glVertex3f(u, v, water_2 + data[xx + (yy + 1) * MESH_DIM] / 100);
			}
			else
			{
				Vec3f normal = computeNormals(xx, yy, water);
				glNormal3f(normal[0], normal[1], normal[2]);
				glVertex3f(u, v, water_1 + data[xx + yy * MESH_DIM] / 100);

				v = (yy + 1) / (float)MESH_DIM;
				v = v * 2.0f - 1.0f;
				normal = computeNormals(xx, yy + 1, water);
				glNormal3f(normal[0], normal[1], normal[2]);
				glVertex3f(u, v, water_2 + data[xx + (yy + 1) * MESH_DIM] / 100);
			}
		}
		glEnd();
	}
	*/

	/*
	for (int yy = 0; yy < MESH_DIM - 1; yy++) {
		//Makes OpenGL draw a triangle at every three consecutive vertices
		glBegin(GL_TRIANGLE_STRIP);
		for (int xx = 0; xx < MESH_DIM; xx++) {
			float u = xx / (float)MESH_DIM;
			float v = yy / (float)MESH_DIM;

			u = u * 2.0f - 1.0f;
			v = v * 2.0f - 1.0f;

			float water_1 = water[xx + yy * MESH_DIM] / 100;
			float water_2 = water[xx + (yy + 1) * MESH_DIM] / 100;
			if ((water_1 * water_2) == 0.0f)
			{
				//glColor4f(0.0f, 0.0f, 1.0f, -1.0f);
			}
			else
			{
				glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
			}
			Vec3f normal = computeNormals(xx, yy, water);
			glNormal3f(normal[0], normal[1], normal[2]);
			glVertex3f(u, v, water_1 + data[xx + yy * MESH_DIM] / 100);

			v = (yy + 1) / (float)MESH_DIM;
			v = v * 2.0f - 1.0f;
			normal = computeNormals(xx, yy + 1, water);
			glNormal3f(normal[0], normal[1], normal[2]);
			glVertex3f(u, v, water_2 + data[xx + (yy + 1) * MESH_DIM] / 100);
		}
		glEnd();
	}
	*/
	
	
	glColor3f(0.5f, 0.35f, 0.05f);
	for (int yy = 0; yy < MESH_DIM - 1; yy++) {
		//Makes OpenGL draw a triangle at every three consecutive vertices
		glBegin(GL_TRIANGLE_STRIP);
		for (int xx = 0; xx < MESH_DIM; xx++) {
			float u = xx / (float)MESH_DIM;
			float v = yy / (float)MESH_DIM;
			u = u * 2.0f - 1.0f;
			v = v * 2.0f - 1.0f;

			Vec3f normal = computeNormals(xx, yy, data);
			glNormal3f(normal[0], normal[1], normal[2]);
			glVertex3f(u, v, data[xx + yy * MESH_DIM] / 100);

			v = (yy + 1) / (float)MESH_DIM;
			v = v * 2.0f - 1.0f;
			normal = computeNormals(xx, yy + 1, data);
			glNormal3f(normal[0], normal[1], normal[2]);
			glVertex3f(u, v, data[xx + (yy + 1) * MESH_DIM] / 100);
		}
		glEnd();
	}

	if (displayWaterFlag)
	{
		glPointSize(3);
		glColor3f(0.0, 0.0, 1.0);
		for (int yy = 0; yy < MESH_DIM - 1; yy++) {
			//Makes OpenGL draw a triangle at every three consecutive vertices
			glBegin(GL_POINTS);
			for (int xx = 0; xx < MESH_DIM - 1; xx++) {

				float u = xx / (float)MESH_DIM;
				float v = yy / (float)MESH_DIM;
				u = u * 2.0f - 1.0f;
				v = v * 2.0f - 1.0f;

				if (water[xx + yy * MESH_DIM] != 0.0f)
				{
					glVertex3f(u, v, (data[xx + yy * MESH_DIM] + water[xx + yy * MESH_DIM]) / 100);
				}
				//printf("%f\n", data[xx + yy * mesh_width].height);
			}
			glEnd();
		}
	}
}


Vec3f computeNormals(int x, int y, float* h)
{
	//Vec3f normals2;
	Vec3f sum(0.0f, 0.0f, 0.0f);

	Vec3f out;
	if (y > 0) {
		//out = Vec3f(0.0f, hs[z - 1][x] - hs[z][x], -1.0f);
		out = Vec3f(0.0f, h[x + (y-1) * MESH_DIM] - h[x + y * MESH_DIM], -1.0f);
	}

	Vec3f in;
	if (y < MESH_DIM - 1) {
		//in = Vec3f(0.0f, hs[z + 1][x] - hs[z][x], 1.0f);
		in = Vec3f(0.0f, h[x + (y + 1) * MESH_DIM] - h[x + y * MESH_DIM], 1.0f);
	}

	Vec3f left;
	if (x > 0) {
		//left = Vec3f(-1.0f, hs[z][x - 1] - hs[z][x], 0.0f);
		left = Vec3f(-1.0f, h[(x - 1) + y * MESH_DIM] - h[x + y * MESH_DIM], 0.0f);
	}

	Vec3f right;
	if (x < MESH_DIM - 1) {
		//right = Vec3f(1.0f, hs[z][x + 1] - hs[z][x], 0.0f);
		right = Vec3f(1.0f, h[(x + 1) + y * MESH_DIM] - h[x + y * MESH_DIM], 0.0f);
	}

	if (x > 0 && y > 0) {
		sum += out.cross(left).normalize();
	}
	if (x > 0 && y < MESH_DIM - 1) {
		sum += left.cross(in).normalize();
	}
	if (x < MESH_DIM - 1 && y < MESH_DIM - 1) {
		sum += in.cross(right).normalize();
	}
	if (x < MESH_DIM - 1 && y > 0) {
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

	if (vbo_height)
	{
		deleteVBO(&vbo_height, cvr_height);
	}

	if (vbo_water)
	{
		deleteVBO(&vbo_water, cvr_water);
	}

	if (vbo_sediment)
	{
		deleteVBO(&vbo_sediment, cvr_sediment);
	}

	if (vbo_new_height)
	{
		deleteVBO(&vbo_new_height, cvr_new_height);
	}

	if (vbo_new_water)
	{
		deleteVBO(&vbo_new_water, cvr_new_water);
	}

	if (vbo_new_sediment)
	{
		deleteVBO(&vbo_new_sediment, cvr_new_sediment);
	}

	if (vbo_rain)
	{
		deleteVBO(&vbo_rain, cvr_rain);
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
		case ('u'):
			rotate_z += 0.8f;
			return;

		case ('w'):
			if (displayWaterFlag) { displayWaterFlag = false;}
			else { displayWaterFlag = true;}
			return;

		case ('r'):
			iterations = 0;
			return;

		case ('p'):
			if (pauseFlag) { pauseFlag = false; }
			else { pauseFlag = true; }
			return;
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


void print(float x, float y, char *string)
{
	glColor3f(1.0f, 1.0f, 1.0f);
	//set the position of the text in the window using the x and y coordinates
	glRasterPos2f(x, y);
	//get the length of the string to display
	int len = (int)strlen(string);

	//loop to display character by character
	for (int i = 0; i < len; i++)
	{
		glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, string[i]);
	}
};

