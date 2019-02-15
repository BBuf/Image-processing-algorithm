#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
#include "algorithm"
#include "vector"
#include "stdio.h"
#include "map"
#include "unordered_map"
#include <math.h>
using namespace std;
using namespace cv;

//Method2
double log2(double N) {
	return log10(N) / log10(2.0);
}

Mat Inrbl(Mat src, double k) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	Mat dsthsi(row, col, CV_64FC3);

	//RGB2HSI
	Mat HH = Mat(row, col, CV_64FC1);
	Mat SS = Mat(row, col, CV_64FC1);
	Mat II = Mat(row, col, CV_64FC1);
	int mp[256] = { 0 };
	double mp2[256] = { 0.0 };
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			double b = src.at<Vec3b>(i, j)[0] / 255.0;
			double g = src.at<Vec3b>(i, j)[1] / 255.0;
			double r = src.at<Vec3b>(i, j)[2] / 255.0;
			double minn = min(b, min(g, r));
			double maxx = max(b, max(g, r));
			double H = 0;
			double S = 0;
			double I = (minn + maxx) / 2.0f;
			if (maxx == minn) {
				dsthsi.at<Vec3f>(i, j)[0] = H;
				dsthsi.at<Vec3f>(i, j)[1] = S;
				dsthsi.at<Vec3f>(i, j)[2] = I;
				HH.at<double>(i, j) = H;
				SS.at<double>(i, j) = S;
				II.at<double>(i, j) = I;
			}
			else {
				double delta = maxx - minn;
				if (I < 0.5) {
					S = delta / (maxx + minn);
				}
				else {
					S = delta / (2.0 - maxx - minn);
				}
				if (r == maxx) {
					if (g > b) {
						H = (g - b) / delta;
					}
					else {
						H = 6.0 + (g - b) / delta;
					}
				}
				else if (g == maxx) {
					H = 2.0 + (b - r) / delta;
				}
				else {
					H = 4.0 + (r - g) / delta;
				}
				H /= 6.0; //除以6，表示在那个部分
				if (H < 0.0)
					H += 1.0;
				if (H > 1)
					H -= 1;
				H = (int)(H * 360); //转成[0, 360]
				dsthsi.at<Vec3f>(i, j)[0] = H;
				dsthsi.at<Vec3f>(i, j)[1] = S;
				dsthsi.at<Vec3f>(i, j)[2] = I;
				HH.at<double>(i, j) = H;
				SS.at<double>(i, j) = S;
				II.at<double>(i, j) = I;
			}
		}
	}
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			mp[(int)((src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2]) / 3)]++;
		}
	}
	//Ostu阈值
	for (int i = 0; i < 256; i++) {
		mp2[i] = (double)mp[i] / (double)(row * col);
	}
	double mI = 0;
	for (int i = 0; i < 256; i++) {
		mI += (i / 255.0)  * mp2[i];
	}
	double var = 0;
	double ThresHold = 0;
	for (int i = 0; i < 256; i++) {
		double T = 1.0 * i / 256;
		double P1 = 0.0;
		double mT = 0.0;
		for (int j = 0; j <= i; j++) {
			P1 += mp2[j];
			mT += (double)(j / 255.0) * mp2[j];
		}
		if (P1 == 0) continue;
		if (((mI*P1 - mT)*(mI*P1 - mT) / (P1*(1 - P1))) > var) {
			var = (mI*P1 - mT)*(mI*P1 - mT) / (P1*(1 - P1));
			ThresHold = T;
		}
	}

	//
	int cnt = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (II.at<double>(i, j) <= ThresHold) {
				cnt++;
			}
		}
	}
	printf("cnt: %d\n", cnt);
	double A = k * sqrt((double)cnt / (double)(row * col - cnt));
	printf("A: %.5f\n", A);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			double D, C;
			if (II.at<double>(i, j) <= ThresHold) {
				D = A;
				C = 1.0 / log2(D + 1.0);
			}
			else {
				D = (double)(ThresHold * A - ThresHold) / double((1 - ThresHold) * (II.at<double>(i, j))) - (double)(ThresHold * A - 1.0) / (1.0 - ThresHold);
				C = 1.0 / log2(D + 1);
			}
			II.at<double>(i, j) = (C * log2(D * (double)II.at<double>(i, j) + 1.0));
		}
	}//

	 //HSI2RGB
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			double r, g, b, M1, M2;
			double H = HH.at<double>(i, j);
			double S = SS.at<double>(i, j);
			double I = II.at<double>(i, j);
			double hue = H / 360;
			if (S == 0) {//灰色
				r = g = b = I;
			}
			else {
				if (I <= 0.5) {
					M2 = I * (1.0 + S);
				}
				else {
					M2 = I + S - I * S;
				}
				M1 = (2.0 * I - M2);
				r = get_Ans(M1, M2, hue + 1.0 / 3.0);
				g = get_Ans(M1, M2, hue);
				b = get_Ans(M1, M2, hue - 1.0 / 3.0);
			}
			dst.at<Vec3b>(i, j)[0] = (int)(b * 255);
			dst.at<Vec3b>(i, j)[1] = (int)(g * 255);
			dst.at<Vec3b>(i, j)[2] = (int)(r * 255);
		}
	}
	return dst;
}

int main() {
	Mat src = cv::imread("G:\\1.jpg");
	Mat dst = Inrbl(src, 50);
	cv::imshow("origin", src);
	cv::imshow("result", dst);
	waitKey(0);
	return 0;
}