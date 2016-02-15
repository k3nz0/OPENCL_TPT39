#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <fstream>
#include <time.h>
#include <math.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <unistd.h>
#include "opencv2/opencv.hpp"

#define STRING_BUFFER_LEN 1024

using namespace cv;
using namespace std;

#define debug(x) cerr << "  - " << #x << ": " << x << endl;
#define debugs(x, y) cerr << "  - " << #x << ": " << x << "\t\t" << #y << ": " << y << endl;
#define debugv(v, n) cerr << #v << " : " << " "; for(int i = 0; i < n; i++) cerr << "\t" << #v << "[" <<  i <<  "]" <<  " : " << v[i] << "\n"; cerr << endl;


void convolve(float *data, float *data_out, int rows, int cols, float *kernel_matrix);
void print_clbuild_errors(cl_program program,cl_device_id device);
unsigned char ** read_file(const char *name);
void GPU_apply_filter(float *input_image_data, float *output_image_data, int rows, int cols, int filter_type);
void CPU_apply_filter(float *input_image_data, float *output_image_data, int rows, int cols, int filter_type);
const char *getErrorString(cl_int error);

void callback(const char *buffer, size_t length, size_t final, void *user_data) {
     fwrite(buffer, 1, length, stdout);
}
void checkError(int status, const char *msg) {
	if(status!=CL_SUCCESS)	
		printf("%s\n",msg);
}

char char_buffer[STRING_BUFFER_LEN];
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_context_properties context_properties[] =
{ 
	CL_CONTEXT_PLATFORM, 0,
	CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
	CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
	0
};
cl_command_queue queue;
cl_program program;
cl_kernel kernel;

cl_mem input_image_data_buf;
cl_mem input_filter_buf;
cl_mem output_image_data_buf;
int status;
cl_int errcode;
  
enum {GAUSSIAN_BLUR, SOBEL_X, SOBEL_Y};

float gaussian_blur[] = {
	1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
	2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
	1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0
};

float alt_sobel_x[] = {
	3, 0, -3,
	10, 0, -10,
	3, 0, -3
};

float alt_sobel_y[] = {
	3, 10, 3,
	0, 0, 0,
	-3, -10, -3
};

void setup_gpu();

#define SHOW
int main(int, char**)
{
	VideoCapture camera("./bourne.mp4");
    if(!camera.isOpened())  // check if we succeeded
        return -1;

    const string NAME = "./output.avi";   // Form the new name with container
    int ex = static_cast<int>(CV_FOURCC('M','J','P','G'));
    Size S = Size((int) camera.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) camera.get(CV_CAP_PROP_FRAME_HEIGHT));
	//Size S = Size(1280,720);
	cout << "SIZE:" << S << endl;
	int ROWS = S.height;
	int COLS = S.width;
	
    VideoWriter outputVideo; // Open the output
    outputVideo.open(NAME, ex, 25, S, true);

    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << NAME << endl;
        return -1;
    }
	time_t start, end;
	double diff, tot = 0;
	int count = 0;
	const char *windowName = "filter";   // Name shown in the GUI window.
    #ifdef SHOW
    	namedWindow(windowName); // Resizable window, might not work on Windows.
    #endif
	

	const int FRAMES = 299;

	float *data_out = (float *) malloc(sizeof(float) * ROWS * COLS);
	float *data_edge_x = (float *) malloc(sizeof(float) * ROWS * COLS);
	float *data_edge_y = (float *) malloc(sizeof(float) * ROWS * COLS);

//----------------------------------------------- SETTING UP GPU ---------------------------------
	setup_gpu();

	input_image_data_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        ROWS * COLS * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_filter_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        3 * 3 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_image_data_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        ROWS * COLS * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");

	// Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_image_data_buf);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_filter_buf);
    checkError(status, "Failed to set argument 2");

	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_image_data_buf);
    checkError(status, "Failed to set argument 3");

    while (true) {
        Mat cameraFrame, displayframe;
		count = count + 1;
		if(count > FRAMES) break;
        camera >> cameraFrame;
        Mat filterframe = Mat(cameraFrame.size(), CV_8UC3);
        Mat grayframe;

		cerr << "FRAME : " << count << endl;
		cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);
    	time (&start);

		/*
			1 - Convert grayframe to 32FC1 (to avoid overflow of int),
			2 - Send grayframe.data to GPU for GaussianBlur x 3 + Scharr x 2 
			2 - Convert back 8UC1

		*/

		grayframe.convertTo(grayframe, CV_32FC1);

		int rows = grayframe.rows;
		int cols = grayframe.cols;
		
		// debugs(rows, cols);

		Mat edge_x = Mat(rows, cols, CV_32FC1),
			edge_y = Mat(rows, cols, CV_32FC1),
			edge, edge_inv;
		
		float *data = (float *) grayframe.data;

		GPU_apply_filter(data, data_out, rows, cols, GAUSSIAN_BLUR);
		memcpy(data, data_out, rows * cols * sizeof(float));
		GPU_apply_filter(data, data_out, rows, cols, GAUSSIAN_BLUR);
		memcpy(data, data_out, rows * cols * sizeof(float));
		GPU_apply_filter(data, data_out, rows, cols, GAUSSIAN_BLUR);
		memcpy(data, data_out, rows * cols * sizeof(float));
		
		GPU_apply_filter(data, data_edge_x, rows, cols, SOBEL_X);
		GPU_apply_filter(data, data_edge_y, rows, cols, SOBEL_Y);

		memcpy(grayframe.data, data, rows * cols * sizeof(float));
		
		edge_x.convertTo(edge_x, CV_32FC1);
		edge_y.convertTo(edge_y, CV_32FC1);

		mempcpy(edge_x.data, data_edge_x, rows * cols * sizeof(float));
		mempcpy(edge_y.data, data_edge_y, rows * cols * sizeof(float));

		grayframe.convertTo(grayframe, CV_8UC1);
		
		edge_x.convertTo(edge_x, CV_8U);
		edge_y.convertTo(edge_y, CV_8U);
		
		addWeighted(edge_x, 0.5, edge_y, 0.5, 0, edge );
        threshold(edge, edge, 80, 255, THRESH_BINARY_INV);

	
		time (&end);
		cvtColor(edge, edge_inv, CV_GRAY2BGR);
    	
		// Clear the output image to black, so that the cartoon line drawings will be black (ie: not drawn).
    	memset((char*)displayframe.data, 0, displayframe.step * displayframe.rows);
		grayframe.copyTo(displayframe, edge);
		
		displayframe.convertTo(displayframe,CV_8UC1);
		cvtColor(displayframe, displayframe, CV_GRAY2BGR);
		outputVideo << displayframe;


	#ifdef SHOW
        imshow(windowName, displayframe);
	#endif

		diff = difftime (end,start);
		tot += diff;
	}
	outputVideo.release();
	camera.release();
	

	clReleaseCommandQueue(queue);
    clReleaseMemObject(input_image_data_buf);
    clReleaseMemObject(input_filter_buf);
    clReleaseMemObject(output_image_data_buf);

	clReleaseProgram(program);
    clReleaseContext(context);
  	
	printf ("FPS %.2lf .\n", (double) FRAMES / tot );

    return EXIT_SUCCESS;

}

void setup_gpu() {
	clGetPlatformIDs(1, &platform, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

    context_properties[1] = (cl_context_properties)platform;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);

    unsigned char **opencl_program = read_file("filters.cl");
    program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
    
    if (program == NULL)
    {
        printf("Program creation failed\n");
        exit(0);
    }	
    
    int success = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(success != CL_SUCCESS) print_clbuild_errors(program,device);
    kernel = clCreateKernel(program, "convolve", NULL);
}


void GPU_apply_filter(float *input_image_data, float *output_image_data, int rows, int cols, int filter_type)
{
	cl_event write_event[3];
	cl_event kernel_event, finish_event;
	float *input_filter = NULL;
	if(filter_type == GAUSSIAN_BLUR) input_filter = gaussian_blur;
	else if(filter_type == SOBEL_X) input_filter = alt_sobel_x;
	else if(filter_type == SOBEL_Y) input_filter = alt_sobel_y;
	else {
		puts("Wrong filter type"); 
		exit(0);
	}

	// Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    status = clEnqueueWriteBuffer(queue, input_image_data_buf, CL_TRUE,
        0, rows * cols * sizeof(float), input_image_data, 0, NULL, &write_event[0]);
    checkError(status, getErrorString(status));

    status = clEnqueueWriteBuffer(queue, input_filter_buf, CL_TRUE,
        0, 3 * 3 * sizeof(float), input_filter, 0, NULL, &write_event[1]);
    checkError(status, getErrorString(status));
	

  	size_t global_work_size[2] = { (size_t) rows, (size_t) cols};
    size_t local_work_size[2] = {1, 1};
    
	status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
            global_work_size, local_work_size, 2, write_event, &kernel_event);
    checkError(status, "Failed to launch kernel 1");

    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue, output_image_data_buf, CL_TRUE,
        0, rows * cols * sizeof(float), output_image_data, 1, &kernel_event, &finish_event);
	checkError(status, "Reading Failure");

	clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
	clReleaseEvent(kernel_event);

}

void CPU_apply_filter(float *input_image_data, float *output_image_data, int rows, int cols, int filter_type) {
	float *input_filter = NULL;
	if(filter_type == GAUSSIAN_BLUR) input_filter = gaussian_blur;
	else if(filter_type == SOBEL_X) input_filter = alt_sobel_x;
	else if(filter_type == SOBEL_Y) input_filter = alt_sobel_y;
	else {
		puts("Wrong filter type"); 
		exit(0);
	}
	
	convolve(input_image_data, output_image_data, rows, cols, input_filter);
}

void convolve(float *data, float *data_out, int rows, int cols, float *kernel_matrix) {
	// int kernel_rows = 3;
	int kernel_cols =  3;

    // dx and dy are useful to go through kernel_matrix
	int dx[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};  
    int dy[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            float sum = 0.0;
            for(int k = 0; k < 9; k++) {
                int x = i + dx[k];
                int y = j + dy[k];
                float d = 0;
                if(x >= 0 && x < rows && y >= 0 && y < cols) {
                    d = data[x * cols + y]; 
                }
                sum += d * kernel_matrix[ (dx[k] + 1) * kernel_cols + (dy[k] + 1)];
            } 
            data_out[i * cols + j] = sum;
        }
    }
}

void print_clbuild_errors(cl_program program,cl_device_id device) {
	cout<<"Program Build failed\n";
	size_t length;
	char buffer[2048];
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
	cout<<"--- Build log ---\n "<<buffer<<endl;
	exit(1);
}

unsigned char ** read_file(const char *name) {
  size_t size;
  unsigned char **output=(unsigned char **)malloc(sizeof(unsigned char *));
  FILE* fp = fopen(name, "rb");
  if (!fp) {
    printf("no such file:%s",name);
    exit(-1);
  }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  *output = (unsigned char *)malloc(size);
  unsigned char **outputstr=(unsigned char **)malloc(sizeof(unsigned char *));
  *outputstr= (unsigned char *)malloc(size);
  if (!*output) {
    fclose(fp);
    printf("mem allocate failure:%s",name);
    exit(-1);
  }

  if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
  fclose(fp);
  printf("file size %d\n",size);
  printf("-------------------------------------------\n");
  snprintf((char *)*outputstr,size,"%s\n",*output);
  printf("%s\n",*outputstr);
  printf("-------------------------------------------\n");
  return outputstr;
}

const char *getErrorString(cl_int error) {
    switch(error){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}