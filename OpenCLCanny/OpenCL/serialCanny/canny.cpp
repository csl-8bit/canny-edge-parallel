#define _USE_MATH_DEFINES 
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define KERNEL_SIZE 3
//#define NUM_GLOBAL_WITEMS 8294400
//#define NUM_GLOBAL_WGROUPS 256
// 0 = GPU , 1 = CPU
#define PLATFORM_SELECTION 1

#include <cmath>
#include "canny.h"
#include <iostream>
#include <math.h>
#include <CL/cl.hpp>
#include <chrono>

using namespace std;

void edges(uint8_t* dst, const uint8_t* src, int weak_threshold, int strong_threshold, int w_, int h_, int NUM_GLOBAL_WGROUPS, int platform) {
	const int NUM_GLOBAL_WITEMS = w_ * h_;
	const double gaussian_kernel[9] = {
		1,2,1,
		2,4,2,
		1,2,1
	};

	const int8_t sobel_kernel_x[] = { -1, 0, 1,
									-2, 0, 2,
									-1, 0, 1 };
	const int8_t sobel_kernel_y[] = { 1, 2, 1,
										 0, 0, 0,
										-1,-2,-1 };

	double* G_ = new double[w_ * h_];
	double* M_ = new double[w_ * h_];
	uint8_t* s_ = new uint8_t[w_ * h_];


	// get all platforms (drivers), e.g. NVIDIA
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);

	if (all_platforms.size() == 0) {
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}

	cl::Platform default_platform = all_platforms[platform];

	// get default device (CPUs, GPUs) of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}

	// use device[1] because that's a CPU; device[0] is the GPU
	cl::Device default_device = all_devices[0];
	cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

	cl::Context context({ default_device });
	cl::Program::Sources sources;

	string kernel_code = "__kernel void apply_gaussian_filter(__global uchar* out_pixels, __global const uchar* in_pixels, int image_width, int image_height, __global double* gaussian_kernel)"
		"{"
		"	const int KERNEL_SIZE = 3;"
		"	int pixNum = get_global_id(0);"
		"	const int offset_xy = ((KERNEL_SIZE - 1) / 2);"
		"	if (!(pixNum >= 0 && pixNum < image_height * image_width))"
		"		return;"
		"	double kernelSum = 0;"
		"	double pixelVal = 0;"
		"	for (int i = 0; i < KERNEL_SIZE; ++i) {"
		"		for (int j = 0; j < KERNEL_SIZE; ++j) {"
		"			if (((pixNum + ((i - offset_xy) * image_width) + j - offset_xy) >= 0)"
		"				&& ((pixNum + ((i - offset_xy) * image_width) + j - offset_xy) <= image_height * image_width - 1)"
		"				&& (((pixNum % image_width) + j - offset_xy) >= 0)"
		"				&& (((pixNum % image_width) + j - offset_xy) <= (image_width - 1))) {"
		"				pixelVal += gaussian_kernel[i * KERNEL_SIZE + j] * in_pixels[pixNum + ((i - offset_xy) * image_width) + j - offset_xy];"
		"				kernelSum += gaussian_kernel[i * KERNEL_SIZE + j];"
		"			}"
		"		}"
		"	}"
		"	out_pixels[pixNum] = (uchar)(pixelVal / kernelSum);"
		"	"
		"}"
		""
		"__kernel void apply_sobel_filter(__global double* gradient_pixels, __global uchar* segment_pixels, __global const uchar* in_pixels, int image_width, int image_height, __global char* sobel_kernel_x, __global char* sobel_kernel_y ) {"
		"	int pixNum = get_global_id(0);"
		"	if (!(pixNum >= 0 && pixNum < image_height * image_width))"
		"		return;"
		"	int x = pixNum % image_width;"
		"	int y = pixNum / image_width;"
		"	int offset_xy = 1; "
		"	if (x < offset_xy || x >= image_width - offset_xy || y < offset_xy || y >= image_height - offset_xy)"
		"		return;"
		"	double convolve_X = 0.0;"
		"	double convolve_Y = 0.0;"
		"	int k = 0;"
		"	int src_pos = x + (y * image_width);"
		"	for (int ky = -offset_xy; ky <= offset_xy; ky++) {"
		"		for (int kx = -offset_xy; kx <= offset_xy; kx++) {"
		"			convolve_X += in_pixels[src_pos + (kx + (ky * image_width))] * sobel_kernel_x[k];"
		"			convolve_Y += in_pixels[src_pos + (kx + (ky * image_width))] * sobel_kernel_y[k];"
		"			k++;"
		"		}"
		"	}"
		"	int segment = 0;"
		"	if (convolve_X == 0.0 || convolve_Y == 0.0) {"
		"		gradient_pixels[src_pos] = 0;"
		"	}"
		"	else {"
		"		gradient_pixels[src_pos] = hypot(convolve_X, convolve_Y);"
		"		double theta = atan2(convolve_Y, convolve_X); "
		"		theta = theta * (360.0 / (2.0 * 3.14159265358979323846264338327950288));"
		"		if ((theta <= 22.5 && theta >= -22.5) || (theta <= -157.5) || (theta >= 157.5))"
		"			segment = 1; "
		"		else if ((theta > 22.5 && theta <= 67.5) || (theta > -157.5 && theta <= -112.5))"
		"			segment = 2; "
		"		else if ((theta > 67.5 && theta <= 112.5) || (theta >= -112.5 && theta < -67.5))"
		"			segment = 3; "
		"		else if ((theta >= -67.5 && theta < -22.5) || (theta > 112.5 && theta < 157.5))"
		"			segment = 4; "
		"	}"
		"	segment_pixels[src_pos] = (uchar)segment;"
		"		"
		"}"
		""
		"__kernel void apply_non_max_suppression(__global double* max_pixels,__global double* gradient_pixels, __global uchar* segment_pixels, int image_width, int image_height) {"
		"	int pos = get_global_id(0);"
		"	if (!(pos >= 0 && pos < image_height * image_width))"
		"		return;"
		"	switch (segment_pixels[pos]) {"
		"	case 1:"
		"		if (segment_pixels[pos - 1] >= gradient_pixels[pos] || gradient_pixels[pos + 1] > gradient_pixels[pos])"
		"			max_pixels[pos] = 0;"
		"		break;"
		"	case 2:"
		"		if (gradient_pixels[pos - (image_width - 1)] >= gradient_pixels[pos] || gradient_pixels[pos + (image_width - 1)] > gradient_pixels[pos])"
		"			max_pixels[pos] = 0;"
		"		break;"
		"	case 3:"
		"		if (gradient_pixels[pos - (image_width)] >= gradient_pixels[pos] || gradient_pixels[pos + (image_width)] > gradient_pixels[pos])"
		"			max_pixels[pos] = 0;"
		"		break;"
		"	case 4:"
		"		if (gradient_pixels[pos - (image_width + 1)] >= gradient_pixels[pos] || gradient_pixels[pos + (image_width + 1)] > gradient_pixels[pos])"
		"			max_pixels[pos] = 0;"
		"		break;"
		"	default:"
		"		max_pixels[pos] = 0;"
		"		break;"
		"	}"
		"}"
		""
		"__kernel void apply_double_threshold(__global uchar* out, __global double* max_pixels, int strong_threshold, int weak_threshold, int image_width, int image_height) {"
		"	int pos = get_global_id(0);"
		"	if (!(pos >= 0 && pos < image_height * image_width))"
		"		return;"
		"	if (max_pixels[pos] > strong_threshold)"
		"		out[pos] = 255;      "
		"	else if (max_pixels[pos] > weak_threshold)"
		"		out[pos] = 100;     "
		"	else"
		"		out[pos] = 0;    "
		"}"
		""
		"__kernel void apply_edge_hysteresis(__global uchar* out, __global uchar* in, int image_width, int image_height) {"
		"	int pos = get_global_id(0);"
		"	if (!(pos >= 0 && pos < image_height * image_width))"
		"		return;"
		"	if (in[pos] == 100) {"
		"		if (in[pos - 1] == 255 || in[pos + 1] == 255 ||"
		"			in[pos - image_width] == 255 || in[pos + image_width] == 255 ||"
		"			in[pos - image_width - 1] == 255 || in[pos - image_width + 1] == 255 ||"
		"			in[pos + image_width - 1] == 255 || in[pos + image_width + 1] == 255)"
		"			out[pos] = 255;"
		"		else"
		"			out[pos] = 0;"
		"	}"
		"}";


	sources.push_back({ kernel_code.c_str(), kernel_code.length() });

	cl::Program program(context, sources);
	if (program.build({ default_device }) != CL_SUCCESS) {
		std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
		exit(1);
	}

	// set up kernels and vectors for GPU code
	cl::CommandQueue queue(context, default_device);
	cl::Kernel apply_gaussian_filter = cl::Kernel(program, "apply_gaussian_filter");
	cl::Kernel apply_sobel_filter = cl::Kernel(program, "apply_sobel_filter");
	cl::Kernel apply_non_max_suppression = cl::Kernel(program, "apply_non_max_suppression");
	cl::Kernel apply_double_threshold = cl::Kernel(program, "apply_double_threshold");
	cl::Kernel apply_edge_hysteresis = cl::Kernel(program, "apply_edge_hysteresis");



	// Get starting timepoint 
	auto start = std::chrono::high_resolution_clock::now();
	// allocate space
	cl::Buffer buffer_dst(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * h_ * w_);
	cl::Buffer buffer_src(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * h_ * w_);
	cl::Buffer buffer_in(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * h_ * w_);
	cl::Buffer buffer_gaussian_kernel(context, CL_MEM_READ_WRITE, sizeof(double) * KERNEL_SIZE * KERNEL_SIZE);
	cl::Buffer buffer_gradient(context, CL_MEM_READ_WRITE, sizeof(double)* h_* w_);
	cl::Buffer buffer_segment(context, CL_MEM_READ_WRITE, sizeof(uint8_t)* h_* w_);
	cl::Buffer buffer_sobel_X(context, CL_MEM_READ_WRITE, sizeof(int8_t) * 3 * 3);
	cl::Buffer buffer_sobel_Y(context, CL_MEM_READ_WRITE, sizeof(int8_t) * 3 * 3);
	cl::Buffer buffer_max_pix(context, CL_MEM_READ_WRITE, sizeof(double) * h_ * w_);

	// push write commands to queue
	queue.enqueueWriteBuffer(buffer_dst, CL_TRUE, 0, sizeof(uint8_t) * h_ * w_, dst);
	queue.enqueueWriteBuffer(buffer_src, CL_TRUE, 0, sizeof(uint8_t) * h_ * w_, src);
	queue.enqueueWriteBuffer(buffer_sobel_X, CL_TRUE, 0, sizeof(int8_t) * 3 * 3, sobel_kernel_x);
	queue.enqueueWriteBuffer(buffer_sobel_Y, CL_TRUE, 0, sizeof(int8_t) * 3 * 3, sobel_kernel_y);
	queue.enqueueWriteBuffer(buffer_gaussian_kernel, CL_TRUE, 0, sizeof(double) * KERNEL_SIZE * KERNEL_SIZE, gaussian_kernel);

	// RUN ZE KERNEL
	apply_gaussian_filter.setArg(0, buffer_dst);
	apply_gaussian_filter.setArg(1, buffer_src);
	apply_gaussian_filter.setArg(2, w_);
	apply_gaussian_filter.setArg(3, h_);
	apply_gaussian_filter.setArg(4, buffer_gaussian_kernel);
	queue.enqueueNDRangeKernel(apply_gaussian_filter, cl::NullRange,  // kernel, offset
		cl::NDRange(NUM_GLOBAL_WITEMS), // global number of work items
		cl::NDRange(NUM_GLOBAL_WGROUPS));               // local number (per group)
	queue.finish();

	// RUN ZE KERNEL
	apply_sobel_filter.setArg(0, buffer_gradient);
	apply_sobel_filter.setArg(1, buffer_segment);
	apply_sobel_filter.setArg(2, buffer_dst);
	apply_sobel_filter.setArg(3, w_);
	apply_sobel_filter.setArg(4, h_);
	apply_sobel_filter.setArg(5, buffer_sobel_X);
	apply_sobel_filter.setArg(6, buffer_sobel_Y);
	queue.enqueueNDRangeKernel(apply_sobel_filter, cl::NullRange,  // kernel, offset
		cl::NDRange(NUM_GLOBAL_WITEMS), // global number of work items
		cl::NDRange(NUM_GLOBAL_WGROUPS));               // local number (per group)
	queue.finish();

	queue.enqueueReadBuffer(buffer_gradient, CL_TRUE, 0, sizeof(double) * h_ * w_, G_);
	queue.enqueueWriteBuffer(buffer_max_pix, CL_TRUE, 0, sizeof(double) * h_ * w_, G_);

	// RUN ZE KERNEL
	apply_non_max_suppression.setArg(0, buffer_max_pix);
	apply_non_max_suppression.setArg(1, buffer_gradient);
	apply_non_max_suppression.setArg(2, buffer_segment);
	apply_non_max_suppression.setArg(3, w_);
	apply_non_max_suppression.setArg(4, h_);
	queue.enqueueNDRangeKernel(apply_non_max_suppression, cl::NullRange,  // kernel, offset
		cl::NDRange(NUM_GLOBAL_WITEMS), // global number of work items
		cl::NDRange(NUM_GLOBAL_WGROUPS));               // local number (per group)
	queue.finish();

	// RUN ZE KERNEL
	apply_double_threshold.setArg(0, buffer_dst);
	apply_double_threshold.setArg(1, buffer_max_pix);
	apply_double_threshold.setArg(2, strong_threshold);
	apply_double_threshold.setArg(3, weak_threshold);
	apply_double_threshold.setArg(4, w_);
	apply_double_threshold.setArg(5, h_);
	queue.enqueueNDRangeKernel(apply_double_threshold, cl::NullRange,  // kernel, offset
		cl::NDRange(NUM_GLOBAL_WITEMS), // global number of work items
		cl::NDRange(NUM_GLOBAL_WGROUPS));               // local number (per group)
	queue.finish();

	queue.enqueueReadBuffer(buffer_dst, CL_TRUE, 0, sizeof(uint8_t) * h_ * w_, dst);
	queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(uint8_t) * h_ * w_, dst);
	// RUN ZE KERNEL
	apply_edge_hysteresis.setArg(0, buffer_dst);
	apply_edge_hysteresis.setArg(1, buffer_in);
	apply_edge_hysteresis.setArg(2, w_);
	apply_edge_hysteresis.setArg(3, h_);
	queue.enqueueNDRangeKernel(apply_edge_hysteresis, cl::NullRange,  // kernel, offset
		cl::NDRange(NUM_GLOBAL_WITEMS),					// global number of work items
		cl::NDRange(NUM_GLOBAL_WGROUPS));               // local number (per group)

	//// read result from GPU to here; including for the sake of timing
	queue.enqueueReadBuffer(buffer_dst, CL_TRUE, 0, sizeof(uint8_t)* h_* w_, dst);
	queue.finish();

	sources.clear();
	delete[] G_;
	delete[] M_;
	delete[] s_;

	// Get ending timepoint 
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	printf("Time taken by OpenCL: %.2f\n",duration.count()/1000.0f);
	//std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;
}