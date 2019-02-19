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
//#pragma omp parallel for num_threads(4)
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
	int pad_x = (pad_Height - row) >> 1;
	int pad_y = (pad_Width - col) >> 1;
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

//pad_A的转换，以适用于openblas，row2col的思想
void convert_A(float *A_convert, const int OutHeight, const int OutWidth, const int pad_Height, const int pad_Width, float *A_pad) {
	for (int i = 0; i < OutHeight; i++) {
		for (int j = 0; j < OutWidth; j++) {
			int index = i * OutHeight * pad_Height + j * pad_Width;
			int col1 = i * pad_Height + j;
			//row2col展开，这里是3*3卷积，展开9次
			A_convert[index] = A_pad[col1];
			A_convert[index + 1] = A_pad[col1 + 1];
			A_convert[index + 2] = A_pad[col1 + 2];

			int col2 = (i + 1) * pad_Height + j;
			A_convert[index + 3] = A_pad[col2];
			A_convert[index + 4] = A_pad[col2 + 1];
			A_convert[index + 5] = A_pad[col2 + 2];

			int col3 = (i + 2) * pad_Height + j;
			A_convert[index + 6] = A_pad[col3];
			A_convert[index + 7] = A_pad[col3 + 1];
			A_convert[index + 8] = A_pad[col3 + 2];
		}
	}
}
//OpenBlas调用sgemm算法
void Matrixmul_blas(const int convAh, const int convAw, float *A_convert, float *B, float *C) {
	const enum CBLAS_ORDER Order = CblasRowMajor;
	const enum CBLAS_TRANSPOSE TransA = CblasNoTrans;
	const enum CBLAS_TRANSPOSE TransB = CblasNoTrans;
	const int M = convAh;//A的行数，C的行数
	const int N = 1;//B的列数，C的列数
	const int K = convAw;//A的列数，B的行数
	const float alpha = 1;
	const float beta = 0;
	const int lda = K;//A的列
	const int ldb = N;//B的列
	const int ldc = N;//C的列

	cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A_convert, lda, B, ldb, beta, C, ldc);
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
	//定义被卷积矩阵宽高
	const int convAw = KernelHeight * KernelWidth;
	const int convAh = OutHeight * OutWidth;
	//转换被卷积矩阵
	float *A_convert = new float[convAh * convAw];
	convert_A(A_convert, OutHeight, OutWidth, pad_Height, pad_Width, A_pad);
	//定义卷积输出矩阵
	float *C = new float[convAh * 1];
	//sgemm算法计算输出矩阵
	Matrixmul_blas(convAh, convAw, A_convert, B, C);
	//输出验证
	Mat dst(OutHeight, OutWidth, CV_32FC1);
	for (int i = 0; i < OutHeight; i++) {
		for (int j = 0; j < OutWidth; j++) {
			dst.at<float>(i, j) = C[i * OutHeight + j];
		}
	}
	cv::imshow("result", dst);
	cv::waitKey(0);
	return 0;
}