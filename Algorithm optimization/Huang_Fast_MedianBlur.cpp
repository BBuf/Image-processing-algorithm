#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

//计算中值
int getMediaValue(const int hist[], int thresh) {
	int sum = 0;
	for (int i = 0; i < 256; i++) {
		sum += hist[i];
		if (sum >= thresh) {
			return i;
		}
	}
	return 255;
}
//快速中值滤波，灰度图
//https://www.cnblogs.com/liulijin/p/9038230.html
Mat fastMedianBlur(Mat src, int diameter) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC1);
	int Hist[256] = { 0 };
	int radius = (diameter - 1) / 2;
	int windowSize = diameter * diameter;
	int threshold = windowSize / 2 + 1;
	uchar *srcData = src.data;
	uchar *dstData = dst.data;
	int right = col - radius;
	int bot = row - radius;
	for (int j = radius; j < bot; j++) {

	}
}