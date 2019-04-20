# Algorithm optimization 优化一些常见的OpenCV算法

- speed_filter2d_in_gray_image.cpp 使用OpenBLAS和OpenMP优化对灰度图的卷积算法(filter2D)
- speed_filter2d_in_rgb_image.cpp 使用OpenBLAS和OpenMP优化对RGB图的卷积算法(filter2D)
- speed_meanFilter_in_gray_image.cpp 使用x86循环展开，openmp优化均值滤波算法
- speed_medianFilter_in_gray_image.cpp AVX和Openmp，优化中值滤波算法
- speed_twoVector_distance.cpp AVX和x86循环展开，优化计算两个向量距离算法
- Huang_Fast_MedianBlur.cpp 利用直方图实现快速中值滤波算法，算法原理：https://blog.csdn.net/just_sort/article/details/87994573
- speed_exp.cpp 在神经网络权值较小的情况下的快速exp算法，算法原理：https://blog.csdn.net/just_sort/article/details/88128200
- fast_meanFilter.cpp 积分图实现o(1)均值滤波，代码来自ImageShop，算法原理：http://www.cnblogs.com/Imageshop/p/6219990.html
- IntegralGraphforMeanFiltering.cpp 积分图实现O(1)均值滤波，算法原理：http://www.cnblogs.com/Imageshop/p/6219990.html
- O(1)_MinMaxFilter.cpp 《STREAMING MAXIMUM-MINIMUM FILTER USING NO MORE THAN THREE COMPARISONS PER ELEMENT
》论文复现，O(1)实现最大最小值滤波算法，算法原理：https://blog.csdn.net/just_sort/article/details/89424709