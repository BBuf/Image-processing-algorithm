#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace cv;
using namespace std;

int rows, cols;
//获取最小值矩阵
int **getMinChannel(cv::Mat img) {
	rows = img.rows;
	cols = img.cols;
	if (img.channels() != 3) {
		fprintf(stderr, "Input Error!");
		exit(-1);
	}
	int **imgGray;
	imgGray = new int *[rows];
	for (int i = 0; i < rows; i++) {
		imgGray[i] = new int[cols];
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int loacalMin = 255;
			for (int k = 0; k < 3; k++) {
				if (img.at<Vec3b>(i, j)[k] < loacalMin) {
					loacalMin = img.at<Vec3b>(i, j)[k];
				}
			}
			imgGray[i][j] = loacalMin;
		}
	}
	return imgGray;
}

//求暗通道
int **getDarkChannel(int **img, int blockSize = 3) {
	if (blockSize % 2 == 0 || blockSize < 3) {
		fprintf(stderr, "blockSize is not odd or too small!");
		exit(-1);
	}
	//计算pool Size
	int poolSize = (blockSize - 1) / 2;
	int newHeight = rows + poolSize - 1;
	int newWidth = cols + poolSize - 1;
	int **imgMiddle;
	imgMiddle = new int *[newHeight];
	for (int i = 0; i < newHeight; i++) {
		imgMiddle[i] = new int[newWidth];
	}
	for (int i = 0; i < newHeight; i++) {
		for (int j = 0; j < newWidth; j++) {
			if (i < rows && j < cols) {
				imgMiddle[i][j] = img[i][j];
			}
			else {
				imgMiddle[i][j] = 255;
			}
		}
	}
	int **imgDark;
	imgDark = new int *[rows];
	for (int i = 0; i < rows; i++) {
		imgDark[i] = new int[cols];
	}
	int localMin = 255;
	for (int i = poolSize; i < newHeight - poolSize; i++) {
		for (int j = poolSize; j < newWidth - poolSize; j++) {
			for (int k = i - poolSize; k < i + poolSize + 1; k++) {
				for (int l = j - poolSize; l < j + poolSize + 1; l++) {
					if (imgMiddle[k][l] < localMin) {
						localMin = imgMiddle[k][l];
					}
				}
			}
			imgDark[i - poolSize][j - poolSize] = localMin;
		}
	}
	return imgDark;
}

Mat MedianFilterFogRemoval(Mat src, float p = 0.95, int KernelSize = 41, int blockSize=3, bool meanModel = false, float percent = 0.001) {
	int row = src.rows;
	int col = src.cols;
	int** imgGray = getMinChannel(src);
	int **imgDark = getDarkChannel(imgGray, blockSize = blockSize);
	//int atmosphericLight = getGlobalAtmosphericLightValue(imgDark, src, meanModel = meanModel, percent = percent);
	int Histgram[256] = { 0 };
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			Histgram[imgDark[i][j]]++;
		}
	}
	int Sum = 0, atmosphericLight = 0;
	for (int i = 255; i >= 0; i--) {
		Sum += Histgram[i];
		if (Sum > row * col * 0.01) {
			atmosphericLight = i;
			break;
		}
	}
	int SumB = 0, SumG = 0, SumR = 0, Amount = 0;
	//printf("%d\n", atmosphericLight);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (imgDark[i][j] >= atmosphericLight) {
				SumB += src.at<Vec3b>(i, j)[0];
				SumG += src.at<Vec3b>(i, j)[1];
				SumR += src.at<Vec3b>(i, j)[2];
				Amount++;
			}
		}
	}
	SumB /= Amount;
	SumG /= Amount;
	SumR /= Amount;
	Mat Filter(row, col, CV_8UC1);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			Filter.at<uchar>(i, j) = imgDark[i][j];
		}
	}
	Mat A(row, col, CV_8UC1);
	medianBlur(Filter, A, KernelSize);
	Mat temp(row, col, CV_8UC1);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Diff = Filter.at<uchar>(i, j) - A.at<uchar>(i, j);
			if (Diff < 0) Diff = -Diff;
			temp.at<uchar>(i, j) = Diff;
		}
	}
	medianBlur(temp, temp, KernelSize);
	Mat B(row, col, CV_8UC1);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Diff = A.at<uchar>(i, j) - temp.at<uchar>(i, j);
			if (Diff < 0) Diff = 0;
			B.at<uchar>(i, j) = Diff;
		}
	}
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Min = B.at<uchar>(i, j) * p;
			if (imgDark[i][j] > Min) {
				B.at<uchar>(i, j) = Min;
			}
			else {
				B.at<uchar>(i, j) = imgDark[i][j];
			}
		}
	}
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int F = B.at<uchar>(i, j);
			int Value;
			if (SumB != F) {
				Value = SumB * (src.at<Vec3b>(i, j)[0] - F) / (SumB - F);
			}
			else {
				Value = src.at<Vec3b>(i, j)[0];
			}
			if (Value < 0) Value = 0;
			else if (Value > 255) Value = 255;
			dst.at<Vec3b>(i, j)[0] = Value;

			if (SumG != F) {
				Value = SumG * (src.at<Vec3b>(i, j)[1] - F) / (SumG - F);
			}
			else {
				Value = src.at<Vec3b>(i, j)[1];
			}
			if (Value < 0) Value = 0;
			else if (Value > 255) Value = 255;
			dst.at<Vec3b>(i, j)[1] = Value;

			if (SumR != F) {
				Value = SumR * (src.at<Vec3b>(i, j)[2] - F) / (SumR - F);
			}
			else {
				Value = src.at<Vec3b>(i, j)[2];
			}
			if (Value < 0) Value = 0;
			else if (Value > 255) Value = 255;
			dst.at<Vec3b>(i, j)[2] = Value;
		}
	}
	return dst;
}


int main() {
	cv::Mat src = cv::imread("F:\\fog\\7.jpg");
	rows = src.rows;
	cols = src.cols;
	cv::Mat dst = MedianFilterFogRemoval(src);
	cv::imshow("origin", src);
	cv::imshow("result", dst);
	cv::imwrite("F:\\fog\\res.jpg", dst);
	waitKey(0);
}