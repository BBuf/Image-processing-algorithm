# OpenCV和C++实现图像处理中各种插值算法

- NearestNeighborInterpolation.cpp C++实现了OpenCV中的最常见的邻近插值。
- BilinearInterpolation.cpp C++实现了OpenCV中的双线性插值，精度比较高，原理请看：https://handspeaker.iteye.com/blog/1545126
- speed_BillnearInterpolation.cpp 使用位运算以及OpenMP指令优化双线性插值代码。
- BicubicInterpolation.cpp C++实现了OpenCV中的双三次插值，同时使用位运算和OpemMP指令优化。原理请看：https://baike.baidu.com/item/%E5%8F%8C%E4%B8%89%E6%AC%A1%E6%8F%92%E5%80%BC/11055947?fr=aladdin