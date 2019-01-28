# OpenCV和C++实现图像处理中各种滤波

- Adaptive Median Filter.cpp C++实现自适应的中值滤波。原理请看：https://www.cnblogs.com/wangguchangqing/p/6379646.html
- Gaussian Filter.cpp C++实现了高斯滤波，支持一个通道和3个通道的高斯滤波。原理请看：https://blog.csdn.net/just_sort/article/details/84305585
- Guided Filter.cpp C++实现了何凯明提出的引导滤波，可以用于暗通道去雾加速。原理请看：https://blog.csdn.net/just_sort/article/details/84324239
- BilateralFilter.cpp C++实现了双边滤波，在滤除噪声的同时可以对原图的边缘细节进行保留。原理请看：https://blog.csdn.net/just_sort/article/details/84957533
- BoxFilter.cpp C++实现了方框滤波(盒滤波)，用于平滑图像。原理为滤波器值全为(1/滤波器元素个数)在原图进行滑动。
