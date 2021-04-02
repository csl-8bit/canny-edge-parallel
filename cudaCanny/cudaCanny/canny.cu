#define _USE_MATH_DEFINES 
#define KERNEL_SIZE 3
#include "canny.h"
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

void apply_canny(uint8_t* final_pixels, const uint8_t* ori_pixels, int weak_threshold, int strong_threshold, int image_width, int image_height, int thd_per_blk) {

	// gaussian kernel
	const double gaussian_kernel[9] = {
		1,2,1,
		2,4,2,
		1,2,1
	};
	const int8_t sobel_kernel_x[] = {   -1, 0, 1,
										-2, 0, 2,
										-1, 0, 1 };
	const int8_t sobel_kernel_y[] = {    1, 2, 1,
										 0, 0, 0,
										-1,-2,-1 };
	/* kernel execution configuration parameters */
	const int num_blks = (image_height * image_width) / thd_per_blk;
	const int grid = 0;

	/* device buffers */
	uint8_t* in, * out;
	double* gradient_pixels;
	double* max_pixels;
	uint8_t* segment_pixels;
	double* gaussian_kernel_gpu;
	int8_t* sobel_kernel_x_gpu;
	int8_t* sobel_kernel_y_gpu;
	uint8_t* final_result;

	float elapsed = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);  //start timer

	/* allocate device memory */
	cudaMalloc((void**)&in, sizeof(uint8_t) * image_height * image_width);
	cudaMalloc((void**)&out, sizeof(uint8_t) * image_height * image_width);
	cudaMalloc((void**)&gradient_pixels, sizeof(double) * image_height * image_width);
	cudaMalloc((void**)&final_result, sizeof(uint8_t) * image_height * image_width);
	cudaMalloc((void**)&max_pixels, sizeof(double) * image_height * image_width);
	cudaMalloc((void**)&segment_pixels, sizeof(uint8_t) * image_height * image_width);
	cudaMalloc((void**)&gaussian_kernel_gpu, sizeof(double) * KERNEL_SIZE * KERNEL_SIZE);
	cudaMalloc((void**)&sobel_kernel_x_gpu, sizeof(int8_t) * 3 * 3);
	cudaMalloc((void**)&sobel_kernel_y_gpu, sizeof(int8_t) * 3 * 3);

	/* data transfer image pixels to device */
	cudaMemcpy(in, ori_pixels, image_height * image_width * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gaussian_kernel_gpu, gaussian_kernel, sizeof(double) * KERNEL_SIZE * KERNEL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(sobel_kernel_x_gpu, sobel_kernel_x, sizeof(int8_t) * KERNEL_SIZE * KERNEL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(sobel_kernel_y_gpu, sobel_kernel_y, sizeof(int8_t) * KERNEL_SIZE * KERNEL_SIZE, cudaMemcpyHostToDevice);

	/* run canny edge detection core - CUDA kernels */
	/* use streams to ensure the kernels are in the same task */
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// 1. gaussian filter
	apply_gaussian_filter << <num_blks, thd_per_blk, grid, stream >> > (out, in, image_width, image_height, gaussian_kernel_gpu);
	// 2.apply sobel kernels
	apply_sobel_filter << <num_blks, thd_per_blk, grid, stream >> > (gradient_pixels, segment_pixels, out,
		image_width, image_height, sobel_kernel_x_gpu, sobel_kernel_y_gpu);
	cudaMemcpy(max_pixels, gradient_pixels, image_height * image_width * sizeof(double), cudaMemcpyDeviceToDevice);
	// 3. local maxima: non maxima suppression
	apply_non_max_suppression << <num_blks, thd_per_blk, grid, stream >> > (max_pixels, gradient_pixels, segment_pixels, image_width, image_height);
	// 4. double threshold
	apply_double_threshold << <num_blks, thd_per_blk, grid, stream >> > (out,max_pixels,strong_threshold,weak_threshold, image_width, image_height);
	// 5. edges with hysteresis
	cudaMemcpy(final_result, out, image_height * image_width * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
	apply_edge_hysteresis << <num_blks, thd_per_blk, grid, stream >> > (final_result, out, image_width, image_height);

	/* wait for everything to finish */
	cudaDeviceSynchronize();

	/* copy result back to the host */
	cudaMemcpy(final_pixels, final_result, image_width * image_height * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	cudaFree(in);
	cudaFree(out);
	cudaFree(gradient_pixels);
	cudaFree(max_pixels);
	cudaFree(segment_pixels);
	cudaFree(gaussian_kernel_gpu);
	cudaFree(sobel_kernel_x_gpu);
	cudaFree(sobel_kernel_y_gpu);
	cudaFree(final_result);

	cudaEventRecord(stop, 0); //end timer
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("The elapsed time in gpu was %.2f ms\n", elapsed);
}

__global__ void apply_gaussian_filter(uint8_t* out_pixels, const uint8_t* in_pixels, int image_width, int image_height, double* gaussian_kernel)
{
	//determine id of thread which corresponds to an individual pixel
	int pixNum = blockIdx.x * blockDim.x + threadIdx.x;
	const int offset_xy = ((KERNEL_SIZE - 1) / 2);
	if (!(pixNum >= 0 && pixNum < image_height * image_width))
		return;

	//Apply Kernel to image
	double kernelSum = 0;
	double pixelVal = 0;
	for (int i = 0; i < KERNEL_SIZE; ++i) {
		for (int j = 0; j < KERNEL_SIZE; ++j) {
			//check edge cases, if within bounds, apply filter
			if (((pixNum + ((i - offset_xy) * image_width) + j - offset_xy) >= 0)
				&& ((pixNum + ((i - offset_xy) * image_width) + j - offset_xy) <= image_height * image_width - 1)
				&& (((pixNum % image_width) + j - offset_xy) >= 0)
				&& (((pixNum % image_width) + j - offset_xy) <= (image_width - 1))) {

				pixelVal += gaussian_kernel[i * KERNEL_SIZE + j] * in_pixels[pixNum + ((i - offset_xy) * image_width) + j - offset_xy];
				kernelSum += gaussian_kernel[i * KERNEL_SIZE + j];
			}
		}
	}
	out_pixels[pixNum] = (uint8_t)(pixelVal / kernelSum);
	
}
__global__ void apply_sobel_filter(double* gradient_pixels, uint8_t* segment_pixels, const uint8_t* in_pixels, int image_width, int image_height, int8_t* sobel_kernel_x, int8_t* sobel_kernel_y ) {
	//Sobel
	int pixNum = blockIdx.x * blockDim.x + threadIdx.x;
	if (!(pixNum >= 0 && pixNum < image_height * image_width))
		return;
	int x = pixNum % image_width;
	int y = pixNum / image_width;
	int offset_xy = 1;  // 3x3
	if (x < offset_xy || x >= image_width - offset_xy || y < offset_xy || y >= image_height - offset_xy)
		return;
	double convolve_X = 0.0;
	double convolve_Y = 0.0;
	int k = 0;
	int src_pos = x + (y * image_width);

	for (int ky = -offset_xy; ky <= offset_xy; ky++) {
		for (int kx = -offset_xy; kx <= offset_xy; kx++) {
			convolve_X += in_pixels[src_pos + (kx + (ky * image_width))] * sobel_kernel_x[k];
			convolve_Y += in_pixels[src_pos + (kx + (ky * image_width))] * sobel_kernel_y[k];
			k++;
		}
	}

	// gradient hypot & direction
	int segment = 0;

	if (convolve_X == 0.0 || convolve_Y == 0.0) {
		gradient_pixels[src_pos] = 0;
	}
	else {
		gradient_pixels[src_pos] = ((std::sqrt((convolve_X * convolve_X) + (convolve_Y * convolve_Y))));
		double theta = std::atan2(convolve_Y, convolve_X);  // radians. atan2 range: -PI,+PI,   // theta : 0 - 2PI
		theta = theta * (360.0 / (2.0 * M_PI));  // degrees

		if ((theta <= 22.5 && theta >= -22.5) || (theta <= -157.5) || (theta >= 157.5))
			segment = 1;  // "-"
		else if ((theta > 22.5 && theta <= 67.5) || (theta > -157.5 && theta <= -112.5))
			segment = 2;  // "/" 
		else if ((theta > 67.5 && theta <= 112.5) || (theta >= -112.5 && theta < -67.5))
			segment = 3;  // "|"
		else if ((theta >= -67.5 && theta < -22.5) || (theta > 112.5 && theta < 157.5))
			segment = 4;  // "\"  
	}
	segment_pixels[src_pos] = (uint8_t)segment;
		
}
__global__ void apply_non_max_suppression(double* max_pixels, double* gradient_pixels, uint8_t* segment_pixels, int image_width, int image_height) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (!(pos >= 0 && pos < image_height * image_width))
		return;
	switch (segment_pixels[pos]) {
	case 1:
		if (segment_pixels[pos - 1] >= gradient_pixels[pos] || gradient_pixels[pos + 1] > gradient_pixels[pos])
			max_pixels[pos] = 0;
		break;
	case 2:
		if (gradient_pixels[pos - (image_width - 1)] >= gradient_pixels[pos] || gradient_pixels[pos + (image_width - 1)] > gradient_pixels[pos])
			max_pixels[pos] = 0;
		break;
	case 3:
		if (gradient_pixels[pos - (image_width)] >= gradient_pixels[pos] || gradient_pixels[pos + (image_width)] > gradient_pixels[pos])
			max_pixels[pos] = 0;
		break;
	case 4:
		if (gradient_pixels[pos - (image_width + 1)] >= gradient_pixels[pos] || gradient_pixels[pos + (image_width + 1)] > gradient_pixels[pos])
			max_pixels[pos] = 0;
		break;
	default:
		max_pixels[pos] = 0;
		break;
	}

}
__global__ void apply_double_threshold(uint8_t* out, double* max_pixels, int strong_threshold, int weak_threshold, int image_width, int image_height) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (!(pos >= 0 && pos < image_height * image_width))
		return;
	if (max_pixels[pos] > strong_threshold)
		out[pos] = 255;      //absolutely edge
	else if (max_pixels[pos] > weak_threshold)
		out[pos] = 100;      //potential edge
	else
		out[pos] = 0;       //absolutely not edge
}
__global__ void apply_edge_hysteresis(uint8_t* out, uint8_t* in, int image_width, int image_height) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (!(pos >= 0 && pos < image_height * image_width))
		return;
	if (in[pos] == 100) {
		if (in[pos - 1] == 255 || in[pos + 1] == 255 ||
			in[pos - image_width] == 255 || in[pos + image_width] == 255 ||
			in[pos - image_width - 1] == 255 || in[pos - image_width + 1] == 255 ||
			in[pos + image_width - 1] == 255 || in[pos + image_width + 1] == 255)
			out[pos] = 255;
		else
			out[pos] = 0;
	}

}

