#include <iostream>
#include <stdio.h>
#include <cblas.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//使用OpenMp和OpenBlas加速filter2D（灰度图）,且卷积核的长宽相等，这里实现的是长宽均为3的卷积核
//RGB转化为灰度图
Mat speed_rgb2gray(Mat src) {
	Mat dst(src.rows, src.cols, CV_8UC1);
#pragma omp parallel for num_threads(4)
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<uchar>(i, j) = ((src.at<Vec3b>(i, j)[0] << 18) + (src.at<Vec3b>(i, j)[0] << 15) + (src.at<Vec3b>(i, j)[0] << 14) +
				(src.at<Vec3b>(i, j)[0] << 11) + (src.at<Vec3b>(i, j)[0] << 7) + (src.at<Vec3b>(i, j)[0] << 7) + (src.at<Vec3b>(i, j)[0] << 5) +
				(src.at<Vec3b>(i, j)[0] << 4) + (src.at<Vec3b>(i, j)[0] << 2) +
				(src.at<Vec3b>(i, j)[1] << 19) + (src.at<Vec3b>(i, j)[1] << 16) + (src.at<Vec3b>(i, j)[1] << 14) + (src.at<Vec3b>(i, j)[1] << 13) +
				(src.at<Vec3b>(i, j)[1] << 10) + (src.at<Vec3b>(i, j)[1] << 8) + (src.at<Vec3b>(i, j)[1] << 4) + (src.at<Vec3b>(i, j)[1] << 3) + (src.at<Vec3b>(i, j)[1] << 1) +
				(src.at<Vec3b>(i, j)[2] << 16) + (src.at<Vec3b>(i, j)[2] << 15) + (src.at<Vec3b>(i, j)[2] << 14) + (src.at<Vec3b>(i, j)[2] << 12) +
				(src.at<Vec3b>(i, j)[2] << 9) + (src.at<Vec3b>(i, j)[2] << 7) + (src.at<Vec3b>(i, j)[2] << 6) + (src.at<Vec3b>(i, j)[2] << 5) + (src.at<Vec3b>(i, j)[2] << 4) + (src.at<Vec3b>(i, j)[2] << 1) >> 20);
		}
	}
	return dst;
}

//A增加Pad的运算
void get_Pad(int pad_Height, int pad_Width, int row, int col, float *A_pad, float *A) {
	int pad_x = pad_Height - row >> 1;
	int pad_y = pad_Width - col >> 1;
	printf("pad_x: %d pad_y: %d\n", pad_x, pad_y);
	for (int i = 0; i < pad_Height; i++) {
		for (int j = 0; j < pad_Width; j++) {
			int index = i * pad_Height + j;
			if (i <= pad_x || i + pad_x > pad_Height) {
				A_pad[index] = 0;
			}
			else {
				if (j <= pad_y || j + pad_y > pad_Width) {
					A_pad[index] = 0;
				}
				else {
					A_pad[index] = A[(i - pad_x) * row + j - pad_y];
				}
			}
		}
	}
}



int main() {
	Mat src = cv::imread("F:\\1.jpg");
	src = speed_rgb2gray(src);
	int row = src.rows;
	int col = src.cols;
	// 将原始的
	float *A = new float[row * col];
	for (int i = 0; i < row * col; i++) {
		int x = i / row;
		int y = i % row;
		A[i] = src.at<float>(x, y);
	}
	const int KernelHeight = 3;
	const int KernelWidth = 3;
	float B[KernelHeight * KernelWidth] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	//卷积核参数初始化为
	const int pad = (KernelHeight - 1) / 2; //需要pad的长度
	const int stride = 1; //卷积核滑动的步长
	//计算卷积输出矩阵的长宽
	const int OutHeight = (row - KernelHeight + 2 * pad) / stride + 1;
	const int OutWidth = (col - KernelWidth + 2 * pad) / stride + 1;
	//计算pad_A
	const int pad_Height = row + 2 * pad;
	const int pad_Width = col + 2 * pad;
	float *A_pad = new float[pad_Height * pad_Width];
	get_Pad(pad_Height, pad_Width, row, col, A_pad, A);

}