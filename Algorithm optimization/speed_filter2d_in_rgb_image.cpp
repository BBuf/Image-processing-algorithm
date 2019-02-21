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

//pad_A的转换，以适用于Openblas
void convert_A(float *A_convert, const int OutHeight, const int OutWidth, const int convAw, const int pad_Height, const int pad_Width, int channel, float *A_pad) {
	int pad_one_channel = pad_Height * pad_Width;
	int seg = channel * convAw;
	for (int c = 0; c < channel; c++) {
		for (int i = 0; i < OutHeight; i++) {
			for (int j = 0; j < OutWidth; j++) {
				int index = c * convAw + i * OutHeight * seg + j * seg;
				int col1 = c*pad_one_channel + i * pad_Height + j;
				A_convert[index] = A_pad[col1];
				A_convert[index + 1] = A_pad[col1 + 1];
				A_convert[index + 2] = A_pad[col1 + 2];

				int col2 = c*pad_one_channel + (i + 1) * pad_Height + j;
				A_convert[index + 3] = A_pad[col2];
				A_convert[index + 4] = A_pad[col2 + 1];
				A_convert[index + 5] = A_pad[col2 + 2];

				int col3 = c*pad_one_channel + (i + 2) * pad_Height + j;
				A_convert[index + 6] = A_pad[col3];
				A_convert[index + 7] = A_pad[col3 + 1];
				A_convert[index + 8] = A_pad[col3 + 2];
			}
		}
	}
}

//kernel 转换以适用opeblas
void convertB(const int convAw, const int channel, float *B, float *B_convert) {
	//3通道3个输出，可以结合https://blog.csdn.net/hai008007/article/details/80209436来理解
	int block_A_convert = convAw * channel;
	for (int c = 0; c < channel; c++) {
		int block = c * block_A_convert;
		for (int i = 0; i < convAw; i++) {
			for (int j = 0; j < channel; j++) {
				if (c == j)
				{
					B_convert[block + i * channel + j] = B[c * convAw + i];
				}
				else
				{
					B_convert[block + i * channel + j] = 0;
				}
			}
		}
	}
}

//OpenBlas矩阵乘法运算
void Matrixmul_blas(const int convAh, const int convAw, float *A_convert, float *B_convert, float *C, const int channel) {
	const enum CBLAS_ORDER Order = CblasRowMajor;
	const enum CBLAS_TRANSPOSE TransA = CblasNoTrans;
	const enum CBLAS_TRANSPOSE TransB = CblasNoTrans;
	const int M = convAh;//A的行数，C的行数
	const int N = channel;//B的列数，C的列数
	const int K = convAw * channel;//A的列数，B的行数
	const float alpha = 1;
	const float beta = 0;
	const int lda = K;//A的列
	const int ldb = N;//B的列
	const int ldc = N;//C的列

	cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A_convert, lda, B_convert, ldb, beta, C, ldc);
}

//将C转换为常用的矩阵排列
void convertC(const int channel, const int convAh, float *C_convert, float *C) {
	for (int c = 0; c < channel; c++) {
		for (int i = 0; i < convAh; i++) {
			C_convert[c*convAh + i] = C[i * channel + c];
		}
	}
}

//验证结果是否正确
void checkResult(const int channel, const int row, const int col, const float *A, const int kernelWidth, const int kernelHeight, const float *B, const int outHeight, const int outWidth, float *C_convert)
{
	cout << "A is:" << endl;
	for (int c = 0; c < channel; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				cout << A[c * row * col + i * row + j] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}

	cout << "B is:" << endl;
	for (int c = 0; c < channel; c++)
	{
		for (int i = 0; i < kernelHeight; i++)
		{
			for (int j = 0; j < kernelWidth; j++)
			{
				cout << B[c * kernelHeight * kernelWidth + i * kernelHeight + j] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}

	cout << "C is:" << endl;
	for (int c = 0; c < channel; c++)
	{
		for (int i = 0; i < outHeight; i++)
		{
			for (int j = 0; j < outWidth; j++)
			{
				cout << C_convert[c * outHeight * outWidth + i * outHeight + j] << " ";
			}
			cout << endl;
		}
		cout << endl;
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
	//定义被卷积矩阵宽高
	const int convAw = KernelHeight * KernelWidth;
	const int convAh = OutHeight * OutWidth;
	//转换被卷积矩阵适应OpenBlas
	float *A_convert = new float[convAh * convAw * channel];
	convert_A(A_convert, OutHeight, OutWidth, convAw, pad_Height, pad_Width, channel, A_pad);
	//转换卷积核适应OpenBlas
	float *B_convert = new float[channel * KernelHeight * KernelWidth * channel];
	convertB(convAw, channel, B, B_convert);
	//定义卷积输出矩阵
	float *C = new float[convAh * channel];
	//cblas计算输出矩阵
	Matrixmul_blas(convAh, convAw, A_convert, B_convert, C, channel);
	//将输出转换为常用的矩阵形式
	float *C_convert = new float[OutHeight * OutWidth * channel];
	convertC(channel, convAh, C_convert, C);
	//输出验证
	checkResult(channel, row, col, A, KernelWidth, KernelHeight, B, OutHeight, OutWidth, C_convert);
	return 0;
}