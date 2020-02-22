#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//计算两个向量pt0->pt1和pt0->pt2的夹角
double angle(Point pt1, Point pt2, Point pt0) {
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

const int N = 5;

int main() {
	// 读入图片
	Mat src = cv::imread("F:\\2stickies.jpg");
	// 记录长宽
	int row = src.rows;
	int col = src.cols;
	// 存储矩阵
	vector <vector<Point> > squares;
	// 将原图转化为灰度图
	Mat gray(row, col, CV_8UC1);
	cvtColor(src, gray, CV_BGRA2GRAY);
	// 中值滤波，半径为9
	medianBlur(gray, gray, 9);
	cv::imshow("median", gray);
	waitKey(0);
	// 存储图像的轮廓
	vector <vector<Point> > contours;
	Mat gray0 = gray.clone();
	for (int l = 0; l < N; l++) {
		if (l == 0) {
			// 先执行Canny边缘检测
			Canny(gray0, gray, 5, 50, 5);
			// 形态学操作
			dilate(gray, gray, Mat(), Point(-1, -1));
		}
		else {
			gray = gray0 >= (l + 1) * 255 / N;
		}
		// 寻找图像的轮廓
		findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
		// 存储多边形的点
		vector <Point> approx;
		for (size_t i = 0; i < contours.size(); i++) {
			approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.02, true);
			if (approx.size() == 4 && fabs(contourArea(Mat(approx))) > 1000 &&
				isContourConvex(Mat(approx))) {
				// 找到多边形最大的角
				double maxCosine = 0;
				for (int j = 2; j < 5; j++) {
					double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
					maxCosine = max(maxCosine, cosine);
				}
				//如果余弦值小于0.3，就判断为矩阵
				if (maxCosine < 0.3) {
					squares.push_back(approx);
				}
			}
		}
	}
	printf("%d\n", squares.size());
	// 把矩形画在原图上
	for (size_t i = 0; i < squares.size(); i++)
	{
		const Point* p = &squares[i][0];
		int n = (int)squares[i].size();
		//dont detect the border
		printf("%d %d\n", p->x, p->y);
		if (p->x > 3 && p->y > 3)
			polylines(src, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
	}
	imshow("result", src);
	waitKey(0);
	return 0;
}