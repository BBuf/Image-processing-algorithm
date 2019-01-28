# 记录一些图像处理中的论文及代码复现

- Retinex MSRCR.cpp 带色彩恢复的多尺度视网膜增强算法。 算法原理请看：http://www.cnblogs.com/Imageshop/archive/2013/04/17/3026881.html
- ImageDataIncrease.cpp 常见的图片数据扩充。包括一些PS算法 具体为旋转，添加高斯，椒盐噪声，增加老照片效果，增加和降低图像饱和度，对原图缩放，亮度增强，对比度增强，磨皮美白，偏色矫正，同态滤波，过曝，灰度化，轮换通道，图像错切，运动模糊，钝化蒙版，PS滤镜算法之球面化 (凸出和凹陷效果)
- HDR.cpp C++复现《Adaptive Local Tone Mapping Based on Retinex for High Dynamic Range Images》， 实现低照度彩色图像恢复。算法原理请看：https://blog.csdn.net/just_sort/article/details/84030723
- Adaptive Logarithmic Mapping For Displaying High Contrast Scenes.cpp C++复现了《Adaptive Logarithmic Mapping For Displaying High Contrast Scenes》 实现低照度彩色图像恢复，效果超棒。算法原理请看：https://blog.csdn.net/just_sort/article/details/84066390
- Single Image Haze Removal Using Dark Channel Prior.cpp C++复现了《Single Image Haze Removal Using Dark Channel Prior》，实现暗通道去雾。算法原理请看：https://blog.csdn.net/just_sort/article/details/84110518
- Local Color Correction.cpp C++复现了《Local Color Correction》论文。算法原理请看：https://blog.csdn.net/just_sort/article/details/84539295
- PartialcolorJudge.cpp C++复现了《基于图像分析的偏色检测及颜色校正方法》论文，实现快速判断图片是否存在偏色。算法原理请看：https://blog.csdn.net/just_sort/article/details/84897976
- Optimized contrast enhancement for real-time image and video dehazin.cpp C++复现了《Optimized contrast enhancement for real-time image and video dehazin》这篇论文，相对于He Kaiming的暗通道去雾，对天空具有天然的免疫力。算法原理请看：https://blog.csdn.net/just_sort/article/details/84932848
- AutoLevelAndAutoContrast.cpp C++复现了自动色阶调整和自动对比度调整，其中自动色阶调整可以用于去雾和水下图像恢复。算法原理请看：https://www.cnblogs.com/Imageshop/archive/2011/11/13/2247614.html
- Contrast Image Correction Method.cpp C++复现了《Contrast Image Correction Method》这篇论文，可以自适应矫正图像。算法原理请看：https://blog.csdn.net/just_sort/article/details/85005510
- MultiScaleDetailBoosting.cpp C++复现了《DARK IMAGE ENHANCEMENT BASED ON PAIRWISE TARGET CONTRAST AND MULTI-SCALE DETAIL BOOSTING》论文，可以用于提升图像不同程度的细节信息。算法原理请看：https://blog.csdn.net/just_sort/article/details/85007555
- Inrbl.cpp C++复现了《改进非线性亮度提升模型的逆光图像恢复》这篇论文，可以做逆光图像恢复。算法原理请看：https://blog.csdn.net/just_sort/article/details/86681325
# ImageFiletering 复现了一些图像滤波算法

# Correction Algorithm 复现了一些图像矫正算法

