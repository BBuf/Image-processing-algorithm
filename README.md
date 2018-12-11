# 记录一些图像处理中的论文及代码复现

- msrcr.cpp 带色彩恢复的多尺度视网膜增强算法 学习链接：http://www.cnblogs.com/Imageshop/archive/2013/04/17/3026881.html
- ImageDataIncrease.cpp 常见的图片数据扩充，包括一些PS算法 具体为旋转，添加高斯，椒盐噪声，增加老照片效果，增加和降低图像饱和度，对原图缩放，亮度增强，对比度增强，磨皮美白，偏色矫正，同态滤波，过曝，灰度化，轮换通道，图像错切，运动模糊，钝化蒙版，PS滤镜算法之球面化 (凸出和凹陷效果)
- HDR.cpp 对《Adaptive Local Tone Mapping Based on Retinex for High Dynamic Range Images》C++朴素实现，无优化，详情请查看https://blog.csdn.net/just_sort/article/details/84030723
- Adaptive Logarithmic Mapping For Displaying High Contrast Scenes.cpp 《Adaptive Logarithmic Mapping For Displaying High Contrast Scenes》c++朴素实现 ,详情请看：https://blog.csdn.net/just_sort/article/details/84066390
- Single Image Haze Removal Using Dark Channel Prior.cpp 《Single Image Haze Removal Using Dark Channel Prior》论文阅读及复现，详情请看：https://blog.csdn.net/just_sort/article/details/84110518
- Local Color Correction.cpp C++复现了《Local Color Correction》论文，详情请参考：https://blog.csdn.net/just_sort/article/details/84539295
- PartialcolorJudge.cpp C++复现了《基于图像分析的偏色检测及颜色校正方法》论文，实现快速判断图片是否存在偏色，详情请看：https://blog.csdn.net/just_sort/article/details/84897976
- Optimized contrast enhancement for real-time image and video dehazin.cpp 复现了《Optimized contrast enhancement for real-time image and video dehazin》这篇论文，相对于He Kaiming的暗通道去雾，对天空具有天然的免疫力。详情请看：https://blog.csdn.net/just_sort/article/details/84932848

# ImageFiletering中记录了一些实现的滤波算法，包括普通的滤波以及自适应滤波算法

