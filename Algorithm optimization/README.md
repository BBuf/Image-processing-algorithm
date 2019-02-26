# Algorithm optimization 优化一些常见的OpenCV算法

- speed_filter2d_in_gray_image.cpp 使用OpenBLAS和OpenMP优化对灰度图的卷积算法(filter2D)
- speed_filter2d_in_rgb_image.cpp 使用OpenBLAS和OpenMP优化对RGB图的卷积算法(filter2D)
- speed_meanFilter_in_gray_image.cpp 使用x86循环展开，openmp优化均值滤波算法

- speed_medianFilter_in_gray_image.cpp AVX和Openmp，优化中值滤波算法