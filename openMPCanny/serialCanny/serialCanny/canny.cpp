#define _USE_MATH_DEFINES 
#define KERNEL_SIZE 3
#include <cmath>
#include "canny.h"
#include <iostream>
#include <math.h>
#include "omp.h"
#include <chrono>
using namespace std::chrono;

void apply_canny(uint8_t* dst, const uint8_t* src, int weak_threshold, int strong_threshold, int image_width, int image_height) {
	double gaussian_kernel[9] = {
		1,2,1,
		2,4,2,
		1,2,1
	};
	omp_set_num_threads(24);
	double* gradient_pixels = new double[image_width * image_height];
	uint8_t* segment_pixels = new uint8_t[image_width * image_height];
	double* matrix_pixels = new double[image_width * image_height];
	uint8_t* double_thres_pixels = new uint8_t[image_width * image_height];

	// 1. gaussian filter
	apply_gaussian_filter(dst, src, image_width, image_height, gaussian_kernel);
	// 2.apply sobel kernels
	apply_sobel_filter(gradient_pixels, segment_pixels, dst, image_width, image_height);
	// 3. local maxima: non maxima suppression
	apply_non_max_suppression(matrix_pixels, gradient_pixels, segment_pixels, image_width, image_height);
	// 4. double threshold
	apply_double_threshold(double_thres_pixels, matrix_pixels, strong_threshold, weak_threshold, image_width, image_height);
	// 5. edges with hysteresis
	apply_edge_hysteresis(dst, double_thres_pixels, image_width, image_height);

	delete[] gradient_pixels;
	delete[] segment_pixels;
	delete[] matrix_pixels;
	delete[] double_thres_pixels;
}


void apply_gaussian_filter(uint8_t* out_pixels, const uint8_t* in_pixels, int image_width, int image_height, double* kernel)
{
	int rows = image_height;
	int cols = image_width;
	const int offset_xy = ((KERNEL_SIZE - 1) / 2);

	//Apply Kernel to image
#pragma omp parallel for
	for (int pixNum = 0; pixNum < image_height * image_width; pixNum++) {
		double kernelSum = 0;
		double pixelVal = 0;
		for (int i = 0; i < KERNEL_SIZE; ++i) {
			for (int j = 0; j < KERNEL_SIZE; ++j) {

				//check edge cases, if within bounds, apply filter
				if (((pixNum + ((i - offset_xy) * cols) + j - offset_xy) >= 0)
					&& ((pixNum + ((i - offset_xy) * cols) + j - offset_xy) <= rows * cols - 1)
					&& (((pixNum % cols) + j - offset_xy) >= 0)
					&& (((pixNum % cols) + j - offset_xy) <= (cols - 1))) {

					pixelVal += kernel[i * KERNEL_SIZE + j] * in_pixels[pixNum + ((i - offset_xy) * cols) + j - offset_xy];
					kernelSum += kernel[i * KERNEL_SIZE + j];
				}
			}
		}
		out_pixels[pixNum] = (uint8_t)(pixelVal / kernelSum);
	}
}
void apply_sobel_filter(double* out_gradient, uint8_t* out_segment, const uint8_t* in, int image_width, int image_height) {
	//Sobel
	const int8_t Gx[] = { -1, 0, 1,
						 -2, 0, 2,
						 -1, 0, 1 };
	const int8_t Gy[] = { 1, 2, 1,
						  0, 0, 0,
						 -1,-2,-1 };
	int offset_xy = 1;  // 3x3
#pragma omp parallel for
	for (int x = offset_xy; x < image_width - offset_xy; x++) {
		for (int y = offset_xy; y < image_height - offset_xy; y++) {
			double convolve_X = 0.0;
			double convolve_Y = 0.0;
			int k = 0;
			int src_pos = x + (y * image_width);

			for (int ky = -offset_xy; ky <= offset_xy; ky++) {
				for (int kx = -offset_xy; kx <= offset_xy; kx++) {
					convolve_X += in[src_pos + (kx + (ky * image_width))] * Gx[k];
					convolve_Y += in[src_pos + (kx + (ky * image_width))] * Gy[k];
					k++;
				}
			}

			// gradient hypot & direction
			int segment = 0;

			if (convolve_X == 0.0 || convolve_Y == 0.0) {
				out_gradient[src_pos] = 0;
			}
			else {
				out_gradient[src_pos] = ((std::sqrt((convolve_X * convolve_X) + (convolve_Y * convolve_Y))));
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
				else
					std::cout << "error " << theta << std::endl;
			}
			out_segment[src_pos] = (uint8_t)segment;
		}
	}
}
void apply_non_max_suppression(double* out_M, double* in_gradient, uint8_t* in_segment, int image_width, int image_height) {
	memcpy(out_M, in_gradient, image_width * image_height * sizeof(double));
#pragma omp parallel for
	for (int x = 1; x < image_width - 1; x++) {
		for (int y = 1; y < image_height - 1; y++) {
			int pos = x + (y * image_width);

			switch (in_segment[pos]) {
			case 1:
				if (in_gradient[pos - 1] >= in_gradient[pos] || in_gradient[pos + 1] > in_gradient[pos])
					out_M[pos] = 0;
				break;
			case 2:
				if (in_gradient[pos - (image_width - 1)] >= in_gradient[pos] || in_gradient[pos + (image_width - 1)] > in_gradient[pos])
					out_M[pos] = 0;
				break;
			case 3:
				if (in_gradient[pos - (image_width)] >= in_gradient[pos] || in_gradient[pos + (image_width)] > in_gradient[pos])
					out_M[pos] = 0;
				break;
			case 4:
				if (in_gradient[pos - (image_width + 1)] >= in_gradient[pos] || in_gradient[pos + (image_width + 1)] > in_gradient[pos])
					out_M[pos] = 0;
				break;
			default:
				out_M[pos] = 0;
				break;
			}
		}
	}
}
void apply_double_threshold(uint8_t* out, double* M_, int strong_threshold, int weak_threshold, int image_width, int image_height) {
#pragma omp parallel for
	for (int x = 0; x < image_width; x++) {
		for (int y = 0; y < image_height; y++) {
			int src_pos = x + (y * image_width);
			if (M_[src_pos] > strong_threshold)
				out[src_pos] = 255;      //absolutely edge
			else if (M_[src_pos] > weak_threshold)
				out[src_pos] = 100;      //potential edge
			else
				out[src_pos] = 0;       //absolutely not edge
		}
	}
}

void apply_edge_hysteresis(uint8_t* dst, uint8_t* in, int image_width, int image_height) {
	memcpy(dst, in, image_width * image_height * sizeof(uint8_t));
#pragma omp parallel for
	for (int x = 1; x < image_width - 1; x++) {
		for (int y = 1; y < image_height - 1; y++) {
			int src_pos = x + (y * image_width);
			if (in[src_pos] == 100) {
				if (in[src_pos - 1] == 255 || in[src_pos + 1] == 255 ||
					in[src_pos - image_width] == 255 || in[src_pos + image_width] == 255 ||
					in[src_pos - image_width - 1] == 255 || in[src_pos - image_width + 1] == 255 ||
					in[src_pos + image_width - 1] == 255 || in[src_pos + image_width + 1] == 255)
					dst[src_pos] = 255;
				else
					dst[src_pos] = 0;
			}
		}
	}
}



