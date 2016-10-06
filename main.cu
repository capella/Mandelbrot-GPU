#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// includes, cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define WIDTH 700*2
#define HEIGHT 500*2

GLuint buffer, pboTextureId;
struct cudaGraphicsResource* buffer_CUDA;

int g_width = WIDTH;
int g_height = HEIGHT;

const int IterationMax=200;
const double EscapeRadius=2;
const double ER2=EscapeRadius*EscapeRadius;
double base_x = -2.5;
double base_y = -2.0;
double zoom = 0.003;

void deleteVBO() { 
    cudaGraphicsUnregisterResource(buffer_CUDA);
    glDeleteBuffers(1, &buffer);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    double tmp, tmp2;
    if (key == GLFW_KEY_D) {
        base_x+=40*zoom;
    } else if (key == GLFW_KEY_A) {
        base_x-=40*zoom;
    } else if (key == GLFW_KEY_W) {
        base_y+=40*zoom;
    } else if (key == GLFW_KEY_S) {
        base_y-=40*zoom;
    } else if (key == GLFW_KEY_I) {
        tmp = g_width * zoom;
        tmp2 = g_height * zoom;
        zoom*=1.6;
        tmp -= g_width * zoom;
        tmp2 -= g_height * zoom;
        base_x += tmp/2.0;
        base_y += tmp2/2.0;
    } else if (key == GLFW_KEY_O) {
        tmp = g_width * zoom;
        tmp2 = g_height * zoom;
        zoom/=1.6;
        tmp -= g_width * zoom;
        tmp2 -= g_height * zoom;
        base_x += tmp/2.0;
        base_y += tmp2/2.0;
    }
}

__global__ void createVertices(float4* positions, double base_x_in, double base_y_in, double zoom_in, unsigned int width) { 


    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate uv coordinates 

    // Write positions 
    int i;
    double Cy = base_y_in + y*zoom_in;
    double Cx = base_x_in + x*zoom_in;

    double Zx = 0.0;
    double Zy = 0.0;
    double Zx2 = Zx*Zx;
    double Zy2 = Zy*Zy;
    /* */
    for (i=0;i<IterationMax && ((Zx2+Zy2)<ER2);i++) {
        Zy=2*Zx*Zy + Cy;
        Zx=Zx2-Zy2 +Cx;
        Zx2=Zx*Zx;
        Zy2=Zy*Zy;
    }
    /* compute  pixel color (24 bit = 3 bytes) */
    if (i==IterationMax) { /*  interior of Mandelbrot set = black */
        positions[y * width + x] = make_float4(0, 0, 0, 1.0f);
    } else { /* exterior of Mandelbrot set = white */
        if (i < 10)
            positions[y * width + x] = make_float4(1, 1, i/10-1, 1.0f);
        else if (i < 30)
            positions[y * width + x] = make_float4((i-10)/20-1, (i-10)/20-1, 1, 1.0f);
        else if (i < 70)
            positions[y * width + x] = make_float4(1, (i-30)/40-1, 1, 1.0f);
        else if (i < 110)
            positions[y * width + x] = make_float4((i-70)/30-1, (i-70)/30-1, 1, 1.0f);
        else if (i < 170)
            positions[y * width + x] = make_float4((i-110)/80-1, 1, 1, 1.0f);
        else
            positions[y * width + x] = make_float4((i-170)/30-1, 1, (i-170)/30-1, 1.0f);
    }



    /*positions[y * width + x] = make_float4(1, 0, 0, 1.0f);*/
}


void create_buffer_and_texture () {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
    unsigned int size = g_width * g_height * 4 * sizeof(float);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&buffer_CUDA, buffer, cudaGraphicsMapFlagsWriteDiscard);
}

void window_size_callback(GLFWwindow* window, int width, int height) {
    g_height = height;
    g_width = width;
    deleteVBO();
    create_buffer_and_texture ();
}

int main(void) {
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(WIDTH/2, HEIGHT/2, "Hello World", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwSetWindowSizeCallback(window, window_size_callback);
    glfwSetKeyCallback(window, key_callback);


    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    cudaSetDevice(0);

    glGenTextures(1, &pboTextureId);
    glBindTexture(GL_TEXTURE_2D, pboTextureId); // <== Bind the texture object!!
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glGenBuffers(1, &buffer);
    create_buffer_and_texture ();

    glActiveTexture(GL_TEXTURE0);
    // This code is using the immediate mode texture object 0. Add an own texture object if needed.
    glBindTexture(GL_TEXTURE_2D, 0); // Just use the immediate mode texture.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE); // Not a texture. default is modulate.


    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window)) {

        float4* positions;
        cudaGraphicsMapResources(1, &buffer_CUDA, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, buffer_CUDA);
        // Execute kernel 
        dim3 dimBlock(16, 16, 1);
        dim3 dimGrid(g_width / dimBlock.x, g_height / dimBlock.y, 1);

        createVertices<<<dimGrid, dimBlock>>>(positions, base_x, base_y, zoom, g_width);
        // Unmap buffer object 
        cudaGraphicsUnmapResources(1, &buffer_CUDA, 0);
        // Render from buffer object 

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_width, g_height, 0, GL_RGBA, GL_FLOAT, NULL);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glEnable(GL_TEXTURE_2D);

        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f);
            glVertex2f(-1.0f, -1.0f);
            glTexCoord2f(1.0f, 0.0f);
            glVertex2f(1.0f, -1.0f);
            glTexCoord2f(1.0f, 1.0f);
            glVertex2f(1.0f, 1.0f);
            glTexCoord2f(0.0f, 1.0f);
            glVertex2f(-1.0f, 1.0f);
        glEnd();

        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}