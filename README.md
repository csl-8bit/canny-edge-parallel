# Parallel Implementation of Canny Edge Detection
A parallel implementation of the Canny Edge Detection algorithm using OpenMP, CUDA and OpenCL.
![640x480](https://user-images.githubusercontent.com/65856165/113412513-10991e80-93eb-11eb-83cc-acfc07fcbec4.jpg)
![640x480](https://user-images.githubusercontent.com/65856165/113412526-155dd280-93eb-11eb-9eda-44f9c56b0101.jpg)

There are 4 Visual Studio 2019 projects with 4 different implementation - OpenMP, CUDA, OpenCL, Serial

# Prerequisite

* CUDA Toolkit 
* OpenMP
* OpenCL
* Visual Studio


# Speedup Result
![image](https://user-images.githubusercontent.com/65856165/113412122-10e4ea00-93ea-11eb-83ab-eccb4a3f4c59.png)

More detailed documentation can be found in canny_doc.pdf

# Limitation and Future works
![image](https://user-images.githubusercontent.com/65856165/113412614-4fc76f80-93eb-11eb-8770-a4386df2fae2.png)

By using Nvidia Visual Profiler, we can see the apply_gaussian_filter kernel takes up 59.1%, and apply_sobel_filter kernel takes up 33.1% of the computation time. Besides, there is no kernel concurrency. We can introduce kernel concurrency by separating the apply_sobel_filter kernel into two kernels, which can be sobel_seperable_pass_x and sobel_seperable_pass_y. These two kernels use two different directions of the Sobel filter. They will have no dependency so they can be executed concurrently. 

In addition, we also found out that Sobel and Gaussian filter is separable functions. In the current implementation, we have not utilised separable filter; therefore, a filter of window size M×M computes M2 operations per pixel. If we utilised separable functions correctly, the cost would be reduced to computing M + M = 2M operations. This is a two step process where the intermediate results from the first separable convolutionis stored and then convolved with the second separable filterto produce the output. We believed the performance would be significantly improved by utilising separable functions.

We can also use multiple streams to parallelise the process of memcpy and kernel execution, although the improvement may not be too significant, it is one thing that can be done in order to push performance to its limit.

# Contribute
* Fork the project.
* Make feature addition or bug fix.
* Send me a pull request.

# License
Copyright (c) 2021, Cheah Siew Lek

The Helicopter Game is provided **as-is** under the **MIT** license. 
For more information see LICENSE.
