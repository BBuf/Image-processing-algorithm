#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
using namespace cv::ml;
using namespace std;

namespace ACE {
	//Gray
	Mat stretchImage(Mat src) {
		int row = src.rows;
		int col = src.cols;
		Mat dst(row, col, CV_64FC1);
		double MaxValue = 0;
		double MinValue = 256.0;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				MaxValue = max(MaxValue, src.at<double>(i, j));
				MinValue = min(MinValue, src.at<double>(i, j));
			}
		}
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				dst.at<double>(i, j) = (1.0 * src.at<double>(i, j) - MinValue) / (MaxValue - MinValue);
				if (dst.at<double>(i, j) > 1.0) {
					dst.at<double>(i, j) = 1.0;
				}
				else if (dst.at<double>(i, j) < 0) {
					dst.at<double>(i, j) = 0;
				}
			}
		}
		return dst;
	}

	Mat getPara(int radius) {
		int size = radius * 2 + 1;
		Mat dst(size, size, CV_64FC1);
		for (int i = -radius; i <= radius; i++) {
			for (int j = -radius; j <= radius; j++) {
				if (i == 0 && j == 0) {
					dst.at<double>(i + radius, j + radius) = 0;
				}
				else {
					dst.at<double>(i + radius, j + radius) = 1.0 / sqrt(i * i + j * j);
				}
			}
		}
		double sum = 0;
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				sum += dst.at<double>(i, j);
			}
		}
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				dst.at<double>(i, j) = dst.at<double>(i, j) / sum;
			}
		}
		return dst;
	}

	Mat NormalACE(Mat src, int ratio, int radius) {
		Mat para = getPara(radius);
		int row = src.rows;
		int col = src.cols;
		int size = 2 * radius + 1;
		Mat Z(row + 2 * radius, col + 2 * radius, CV_64FC1);
		for (int i = 0; i < Z.rows; i++) {
			for (int j = 0; j < Z.cols; j++) {
				if((i - radius >= 0) && (i - radius < row) && (j - radius >= 0) && (j - radius < col)) {
					Z.at<double>(i, j) = src.at<double>(i - radius, j - radius);
				}
				else {
					Z.at<double>(i, j) = 0;
				}
			}
		}

		Mat dst(row, col, CV_64FC1);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				dst.at<double>(i, j) = 0.f;
			}
		}
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (para.at<double>(i, j) == 0) continue;
				for (int x = 0; x < row; x++) {
					for (int y = 0; y < col; y++) {
						double sub = src.at<double>(x, y) - Z.at<double>(x + i, y + j);
						double tmp = sub * ratio;
						if (tmp > 1.0) tmp = 1.0;
						if (tmp < -1.0) tmp = -1.0;
						dst.at<double>(x, y) += tmp * para.at<double>(i, j);
					}
				}
			}
		}
		return dst;
	}

	Mat FastACE(Mat src, int ratio, int radius) {
		int row = src.rows;
		int col = src.cols;
		if (min(row, col) <= 2) {
			Mat dst(row, col, CV_64FC1);
			for (int i = 0; i < row; i++) {
				for (int j = 0; j < col; j++) {
					dst.at<double>(i, j) = 0.5;
				}
			}
			return dst;
		}
		
		Mat Rs((row + 1) / 2, (col + 1) / 2, CV_64FC1);
		
		resize(src, Rs, Size((col + 1) / 2, (row + 1) / 2));
		Mat Rf= FastACE(Rs, ratio, radius);
		resize(Rf, Rf, Size(col, row));
		resize(Rs, Rs, Size(col, row));
		Mat dst(row, col, CV_64FC1);
		Mat dst1 = NormalACE(src, ratio, radius);
		Mat dst2 = NormalACE(Rs, ratio, radius);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				dst.at<double>(i, j) = Rf.at<double>(i, j) + dst1.at<double>(i, j) - dst2.at<double>(i, j);
			}
		}
		return dst;
	}

	Mat getACE(Mat src, int ratio, int radius) {
		int row = src.rows;
		int col = src.cols;
		vector <Mat> v;
		split(src, v);
		v[0].convertTo(v[0], CV_64FC1);
		v[1].convertTo(v[1], CV_64FC1);
		v[2].convertTo(v[2], CV_64FC1);
		Mat src1(row, col, CV_64FC1);
		Mat src2(row, col, CV_64FC1);
		Mat src3(row, col, CV_64FC1);

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				src1.at<double>(i, j) = 1.0 * src.at<Vec3b>(i, j)[0] / 255.0;
				src2.at<double>(i, j) = 1.0 * src.at<Vec3b>(i, j)[1] / 255.0;
				src3.at<double>(i, j) = 1.0 * src.at<Vec3b>(i, j)[2] / 255.0;
			}
		}
		src1 = stretchImage(FastACE(src1, ratio, radius));
		src2 = stretchImage(FastACE(src2, ratio, radius));
		src3 = stretchImage(FastACE(src3, ratio, radius));

		Mat dst1(row, col, CV_8UC1);
		Mat dst2(row, col, CV_8UC1);
		Mat dst3(row, col, CV_8UC1);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				dst1.at<uchar>(i, j) = (int)(src1.at<double>(i, j) * 255);
				if (dst1.at<uchar>(i, j) > 255) dst1.at<uchar>(i, j) = 255;
				else if (dst1.at<uchar>(i, j) < 0) dst1.at<uchar>(i, j) = 0;
				dst2.at<uchar>(i, j) = (int)(src2.at<double>(i, j) * 255);
				if (dst2.at<uchar>(i, j) > 255) dst2.at<uchar>(i, j) = 255;
				else if (dst2.at<uchar>(i, j) < 0) dst2.at<uchar>(i, j) = 0;
				dst3.at<uchar>(i, j) = (int)(src3.at<double>(i, j) * 255);
				if (dst3.at<uchar>(i, j) > 255) dst3.at<uchar>(i, j) = 255;
				else if (dst3.at<uchar>(i, j) < 0) dst3.at<uchar>(i, j) = 0;
			}
		}
		vector <Mat> out;
		out.push_back(dst1);
		out.push_back(dst2);
		out.push_back(dst3);
		Mat dst;
		merge(out, dst);
		return dst;
	}
}

using namespace ACE;

int main() {
	Mat src = imread("F:\\sky.jpg");
	Mat dst = getACE(src, 4, 7);
	imshow("origin", src);
	imshow("result", dst);
	waitKey(0);
}