// OpenCVTest.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include <iostream>
#include <stdio.h>
#include <cblas.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//使用OpenMp和OpenBlas加速filter2D(RGB度图）,且卷积核的长宽相等，这里实现的是长宽均为3且通道数为3的卷积核(相当于Tensor)
//A加pad的运算
void get_Pad(int pad_Height, int pad_Width, int row, int col, int channel, float *A_pad, const float *A) {
	int pad_x = (pad_Height - row) >> 1;
	int pad_y = (pad_Width - col) >> 1;
	printf("pad_x: %d pad_y: %d\n", pad_x, pad_y);
	const int pad_one_channel = pad_Height * pad_Width;
	const int org_one_channel = row * col;
	for (int c = 0; c < channel; c++) {
		for (int i = 0; i < pad_Height; i++) {
			for (int j = 0; j < pad_Width; j++) {
				int index = c * pad_one_channel + i * pad_Height + j;
				if (i <= pad_x || i + pad_x > pad_Height) {
					A_pad[index] = 0;
				}
				else {
					if (j <= pad_y || j + pad_y > pad_Width) {
						A_pad[index] = 0;
					}
					else {
						A_pad[index] = A[c * org_one_channel + (i - 1) * row + j - 1];
					}
				}
			}
		}
	}
}


int main() {
	//RGB图
	Mat src = cv::imread("F:\\1.jpg");
	int row = src.rows;
	int col = src.cols;
	int channel = src.channels();
	float *A = new float[row * col * channel];
	int cnt = 0;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				A[cnt++] = 1.0 * src.at<Vec3b>(i, j)[k];
			}
		}
	}
	const int KernelHeight = 3;
	const int KernelWidth = 3;
	float *B = new float[KernelHeight * KernelWidth * channel];
	cnt = 0;
	//这里随便赋值测试
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < KernelHeight; i++) {
			for (int j = 0; j < KernelWidth; j++) {
				B[cnt++] = k + 1;
			}
		}
	}
	//卷积核参数初始化为
	const int pad = (KernelHeight - 1) / 2; //需要pad的长度
	const int stride = 1; //卷积核滑动的步长
	const int OutHeight = (row - KernelHeight + 2 * pad) / stride + 1;
	const int OutWidth = (col - KernelWidth + 2 * pad) / stride + 1;
	//计算3维pad_A
	const int pad_Height = row + 2 * pad;
	const int pad_Width = col + 2 * pad;
	float *A_pad = new float[pad_Height * pad_Width * channel];
	get_Pad(pad_Height, pad_Width, row, col, channel, A_pad, A);
	//
}