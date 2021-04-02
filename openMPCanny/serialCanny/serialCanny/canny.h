#include <iostream>
void apply_canny(uint8_t* dst, const uint8_t* src, int weak_threshold, int strong_threshold, int w_, int h_);
void apply_gaussian_filter(uint8_t* dst, const uint8_t* src, int image_width, int image_height, double* in_kernel);
void apply_sobel_filter(double* out_gradient, uint8_t* out_segment, const uint8_t* in, int image_width, int image_height);
void apply_non_max_suppression(double* out_M, double* in_gradient, uint8_t* in_segment, int image_width, int image_height);
void apply_double_threshold(uint8_t* dst, double* M_, int strong_threshold, int weak_threshold, int image_width, int image_height);
void apply_edge_hysteresis(uint8_t* dst, uint8_t* in, int image_width, int image_height);


