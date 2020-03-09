#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
//针对灰度图的均值滤波+CVPR 2019的SideWindowFilter
//其他种类的滤波直接换核即可

int cnt[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
vector <int> filter[8];

void InitFilter(int radius) {
	int n = radius * 2 + 1;
	for (int i = 0; i < 8; i++) {
		cnt[i] = 0;
		filter[i].clear();
	}
	for (int i = 0; i < 8; i++) {
		for (int x = 0; x < n; x++) {
			for (int y = 0; y < n; y++) {
				if (i == 0 && x <= radius && y <= radius) {
					filter[i].push_back(1);
				}
				else if (i == 1 && x <= radius && y >= radius) {
					filter[i].push_back(1);
				}
				else if (i == 2 && x >= radius && y <= radius) {
					filter[i].push_back(1);
				}
				else if (i == 3 && x >= radius && y >= radius) {
					filter[i].push_back(1);
				}
				else if (i == 4 && x <= radius) {
					filter[i].push_back(1);
				}
				else if (i == 5 && x >= radius) {
					filter[i].push_back(1);
				}
				else if (i == 6 && y >= radius) {
					filter[i].push_back(1);
				}
				else if (i == 7 && y <= radius) {
					filter[i].push_back(1);
				}
				else {
					filter[i].push_back(0);
				}
			}
		}
	}
	for (int i = 0; i < 8; i++) {
		int sum = 0;
		for (int j = 0; j < filter[i].size(); j++) sum += filter[i][j] == 1;
		cnt[i] = sum;
	}
}

Mat SideWindowFilter(Mat src, int radius = 1) {
	int row = src.rows;
	int col = src.cols;
	int channels = src.channels();
	InitFilter(radius);
	for (int i = 0; i < 8; i++) {
		printf("%d ", cnt[i]);
	}
	printf("\n");
	if (channels == 1) {
		Mat dst(row, col, CV_8UC1);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (i < radius || i + radius >= row || j < radius || j + radius >= col) {
					dst.at<uchar>(i, j) = src.at<uchar>(i, j);
					continue;
				}
				int minn = 256;
				int pos = 0;
				for (int k = 0; k < 8; k++) {
					int val = 0;
					int id = 0;
					for (int x = -radius; x <= radius; x++) {
						for (int y = -radius; y <= radius; y++) {
							//if (x == 0 && y == 0) continue;
							val += src.at<uchar>(i + x, j + y) * filter[k][id++];
						}
					}
					val /= cnt[k];
					if (abs(val - src.at<uchar>(i, j)) < minn) {
						minn = abs(val - src.at<uchar>(i, j));
						pos = k;
					}
				}
				int val = 0;
				int id = 0;
				for (int x = -radius; x <= radius; x++) {
					for (int y = -radius; y <= radius; y++) {
						//if (x == 0 && y == 0) continue;
						val += src.at<uchar>(i + x, j + y) * filter[pos][id++];
					}
				}
				dst.at<uchar>(i, j) = val / cnt[pos];
			}
		}
		return dst;
	}
	Mat dst(row, col, CV_8UC3);
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (i < radius || i + radius >= row || j < radius || j + radius >= col) {
					dst.at<Vec3b>(i, j)[c] = src.at<Vec3b>(i, j)[c];
					continue;
				}
				int minn = 256;
				int pos = 0;
				for (int k = 0; k < 8; k++) {
					int val = 0;
					int id = 0;
					for (int x = -radius; x <= radius; x++) {
						for (int y = -radius; y <= radius; y++) {
							//if (x == 0 && y == 0) continue;
							val += src.at<Vec3b>(i + x, j + y)[c] * filter[k][id++];
						}
					}
					val /= cnt[k];
					if (abs(val - src.at<Vec3b>(i, j)[c]) < minn) {
						minn = abs(val - src.at<Vec3b>(i, j)[c]);
						pos = k;
					}
				}
				int val = 0;
				int id = 0;
				for (int x = -radius; x <= radius; x++) {
					for (int y = -radius; y <= radius; y++) {
						//if (x == 0 && y == 0) continue;
						val += src.at<Vec3b>(i + x, j + y)[c] * filter[pos][id++];
					}
				}
				dst.at<Vec3b>(i, j)[c] = val / cnt[pos];
			}
		}
	}
	return dst;
}

int main() {
	Mat src = imread("F:\\panda.jpg");
	cv::imshow("origin", src);

	for (int i = 0; i < 9; i++) {
		src = SideWindowFilter(src, 3);
		//medianBlur(src, src, 3);
	}
	//Mat dst;
	//medianBlur(src, dst, 3);
	Mat dst = SideWindowFilter(src, 3);
	imshow("result", dst);
	imwrite("F:\\res.jpg", dst);
	waitKey(0);
}