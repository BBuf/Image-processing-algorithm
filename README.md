# 本工程记录一些图像处理中的论文复现及数字图像处理知识点

## 1. Correction Algorithm 复现了一些图像矫正算法

## 2. ImageFiletering 复现了一些图像滤波算法

## 3. Feature Extraction 复现了一些图像特征提取算法

## 4. License Plate Recognition System 实现车牌号码识别算法

## 5. Color Space Conversion 实现和优化各种色彩空间转换算法

## 6. Algorithm optimization 优化一些常见的OpenCV算法

## 7.PhotoShop Algorithm 破解一些PhotoShop算法

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
- unevenLightCompensate.cpp C++复现了《一种基于亮度均衡的图像阈值分割技术》这篇论文的光照补偿部分，可以对光照不均匀，曝光，逆光图像做亮度均衡，效果不错。原理请看：https://blog.csdn.net/just_sort/article/details/88551771
- Adaptive correction algorithm for illumination inhomogeneity image based on two-dimensional gamma function.cpp C++复现了《基于二维伽马函数的光照不均匀图像自适应校正算法》这篇论文，对光照不均匀的图像有较好的校正效果，且不会像Retiex那样出现光晕。原理请看：https://blog.csdn.net/just_sort/article/details/88569129
- Real-time adaptive contrast enhancement for imaging sensors.cpp C++复现了《Real-time adaptive contrast enhancement for imaging sensors》这篇论文，实时自适应局部对比度增强算法。原理请看：https://blog.csdn.net/just_sort/article/details/85208124
- AutomaticWhiteBalanceMethod.cpp C++复现了《A Novel Automatic White Balance Method For Digital Still Cameras》这篇论文，实现了效果比完美反射更好得白平衡效果。原理请看：https://blog.csdn.net/just_sort/article/details/89183909
- Automatic Color Equalization(ACE) and its Fast Implementation.cpp C++复现了IPOL《Automatic Color  Equalization(ACE) and its Fast Implementation》论文，用于自动色彩均衡。原理请看：https://blog.csdn.net/just_sort/article/details/85237711
- Single Image Haze Removal Using Dark Channel Prior(Guided Filter).cpp C++复现了《Single Image Haze Removal Using Dark Channel Prior》论文，同时使用了何博士提到导向滤波来估计透射率，比原始实现效果更好。算法原理：https://blog.csdn.net/just_sort/article/details/89470403
- MedianFilterFogRemoval.cpp C++复现了《[一种单幅图像去雾方法](http://wenku.baidu.com/link?url=ZoNmd4noFbWZOGKCHus4anP83t8gcc0xWDu9QCfgQuzwn7LxUoBbZmMxrUAFYM3_YEMoQH3DdvYD8j1hdcHt5Wz4LhdvDe4_GZYXrqCYco3)》使用中值滤波进行去雾，原理请看：https://blog.csdn.net/just_sort/article/details/89520776
- FastDefoggingBasedOnSingleImage.cpp C++复现了《基于单幅图像的快速去雾》论文，原理请看：https://blog.csdn.net/just_sort/article/details/90205686
- BoxSideWindowFilter.cpp C++复现了CVPR2019《Side Window Filter》论文(Box Filter)，实现霸气的强制保边，原理请看：https://blog.csdn.net/just_sort/article/details/93664078
- MedianSideWindowFilter.cpp C++复现了CVPR2019《Side Window Filter》论文(Median Filter)，实现霸气的强制保边，原理请看：https://blog.csdn.net/just_sort/article/details/93664078
- RectangleDetection.cpp C++复现了StackOverFlow上面的一个有趣的矩形检测算法，并且配合Side Window Filter可以取得更好的效果，原理请看：https://blog.csdn.net/just_sort/article/details/104754937





![我的公众号，欢迎关注](image/weixin.jpg)

