#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace cv;
using namespace std;

int getMax(Mat src) {
	int row = src.rows;
	int col = src.cols;
	int temp = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			temp = max((int)src.at<uchar>(i, j), temp);
		}
		if (temp == 255) return temp;
	}
	return temp;
}

Mat dehaze(Mat src) {
	double eps;
	int row = src.rows;
	int col = src.cols;
	Mat M = Mat::zeros(row, col, CV_8UC1);
	Mat M_max = Mat::zeros(row, col, CV_8UC1);
	Mat M_ave = Mat::zeros(row, col, CV_8UC1);
	Mat L = Mat::zeros(row, col, CV_8UC1);
	Mat dst = Mat::zeros(row, col, CV_8UC3);
	double m_av, A;
	//get M
	double sum = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			uchar r, g, b, temp1, temp2;
			b = src.at<Vec3b>(i, j)[0];
			g = src.at<Vec3b>(i, j)[1];
			r = src.at<Vec3b>(i, j)[2];
			temp1 = min(min(r, g), b);
			temp2 = max(max(r, g), b);
			M.at<uchar>(i, j) = temp1;
			M_max.at<uchar>(i, j) = temp2;
			sum += temp1;
		}
	}
	m_av = sum / (row * col * 255);
	eps = 0.85 / m_av;
	boxFilter(M, M_ave, CV_8UC1, Size(51, 51));
	double delta = min(0.9, eps*m_av);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			L.at<uchar>(i, j) = min((int)(delta * M_ave.at<uchar>(i, j)), (int)M.at<uchar>(i, j));
		}
	}
	A = (getMax(M_max) + getMax(M_ave)) * 0.5;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int temp = L.at<uchar>(i, j);
			for (int k = 0; k < 3; k++) {
				int val = A * (src.at<Vec3b>(i, j)[k] - temp) / (A - temp);
				if (val > 255) val = 255;
				if (val < 0) val = 0;
				dst.at<Vec3b>(i, j)[k] = val;
			}
		}
	}
	return dst;
}

int main() {
	Mat src = imread("F:\\fog\\19.jpg");
	Mat dst = dehaze(src);
	cv::imshow("origin", src);
	cv::imshow("result", dst);
	//cv::imwrite("F:\\fog\\res.jpg", dst);
	waitKey(0);
	return 0;
}