#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
#include "algorithm"
#include "vector"
#include "stdio.h"
#include "map"
#include "unordered_map"
#include <math.h>
#include <ctime>

using namespace std;
using namespace cv;

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

Mat WaterSegment(Mat src) {
	int row = src.rows;
	int col = src.cols;
	//1. 将RGB图像灰度化
	Mat grayImage = speed_rgb2gray(src);
	//2. 使用大津法转为二值图，并做形态学闭合操作
	threshold(grayImage, grayImage, 0, 255, THRESH_BINARY | THRESH_OTSU);
	//3. 形态学闭操作
	Mat kernel = getStructuringElement(MORPH_RECT, Size(9, 9), Point(-1, -1));
	morphologyEx(grayImage, grayImage, MORPH_CLOSE, kernel);
	//4. 距离变换
	distanceTransform(grayImage, grayImage, DIST_L2, DIST_MASK_3, 5);
    //5. 将图像归一化到[0, 1]范围
	normalize(grayImage, grayImage, 0, 1, NORM_MINMAX);
	//6. 将图像取值范围变为8位(0-255)
	grayImage.convertTo(grayImage, CV_8UC1);
	//7. 再使用大津法转为二值图，并做形态学闭合操作
	threshold(grayImage, grayImage, 0, 255, THRESH_BINARY | THRESH_OTSU);
	morphologyEx(grayImage, grayImage, MORPH_CLOSE, kernel);
	//8. 使用findContours寻找marks
	vector<vector<Point>> contours;
	findContours(grayImage, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(-1, -1));
	Mat marks;
	for (size_t i = 0; i < contours.size(); i++)
	{
		//static_cast<int>(i+1)是为了分水岭的标记不同，区域1、2、3...这样才能分割
		drawContours(marks, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i + 1)), 2);
	}
	//9. 对原图做形态学的腐蚀操作
	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	morphologyEx(src, src, MORPH_ERODE, k);
	//10. 调用opencv的分水岭算法
	watershed(src, marks);
	//11. 随机分配颜色
	vector<Vec3b> colors;
	for (size_t i = 0; i < contours.size(); i++) {
		int r = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int b = theRNG().uniform(0, 255);
		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}

	// 12. 显示
	Mat dst = Mat::zeros(marks.size(), CV_8UC3);
	int index = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			index = marks.at<int>(i, j);
			if (index > 0 && index <= contours.size()) {
				dst.at<Vec3b>(i, j) = colors[index - 1];
			}
			else if (index == -1)
			{
				dst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else {
				dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
		}
	}
	return dst;
}

int main() {
	Mat src = cv::imread("F:\\1.jpg");
	Mat dst = WaterSegment(src);
	cv::imshow("result", dst);
	cv::imwrite("F:\\2.jpg", dst);
	waitKey(0);
	return 0;
}
