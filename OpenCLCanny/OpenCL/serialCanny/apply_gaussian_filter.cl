__kernel void apply_sobel_filter(__global double* gradient_pixels, __global uchar* segment_pixels, __global const uchar* in_pixels, int image_width, int image_height, __global uchar* sobel_kernel_x, __global uchar* sobel_kernel_y ) {

	int rows = get_global_id(0);
	int cols = get_global_id(1);
	int pixNum = rows * image_height + cols;
	if (!(pixNum >= 0 && pixNum < image_height * image_width))
		return;
	int x = pixNum % image_width;
	int y = pixNum / image_width;
	int offset_xy = 1; 
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

	int segment = 0;

	if (convolve_X == 0.0 || convolve_Y == 0.0) {
		gradient_pixels[src_pos] = 0;
	}
	else {
		gradient_pixels[src_pos] = ((std::sqrt((convolve_X * convolve_X) + (convolve_Y * convolve_Y))));
		double theta = std::atan2(convolve_Y, convolve_X); 
		theta = theta * (360.0 / (2.0 * M_PI));

		if ((theta <= 22.5 && theta >= -22.5) || (theta <= -157.5) || (theta >= 157.5))
			segment = 1; 
		else if ((theta > 22.5 && theta <= 67.5) || (theta > -157.5 && theta <= -112.5))
			segment = 2; 
		else if ((theta > 67.5 && theta <= 112.5) || (theta >= -112.5 && theta < -67.5))
			segment = 3; 
		else if ((theta >= -67.5 && theta < -22.5) || (theta > 112.5 && theta < 157.5))
			segment = 4; 
	}
	segment_pixels[src_pos] = (uchar)segment;
		
}