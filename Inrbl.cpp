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
#define PI 3.1415926
double log2(double N) {
	return log10(N) / log10(2.0);
}

Mat Inrbl(Mat src, double k) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_64FC3);
	Mat dsthsi(row, col, CV_64FC3);

	//RGB2HSI
	Mat H = Mat(row, col, CV_64FC1);
	Mat S = Mat(row, col, CV_64FC1);
	Mat I = Mat(row, col, CV_64FC1);
	int mp[256] = { 0 };
	double mp2[256] = { 0.0 };
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			double h, s, newi, th;
			double B = (double)src.at<Vec3b>(i, j)[0] / 255.0;
			double G = (double)src.at<Vec3b>(i, j)[1] / 255.0;
			double R = (double)src.at<Vec3b>(i, j)[2] / 255.0;
			double mi, mx;
			if (R > G && R > B) {
				mx = R;
				mi = min(G, B);
			}
			else {
				if (G > B) {
					mx = G;
					mi = min(R, B);
				}
				else {
					mx = B;
					mi = min(R, G);
				}
			}
			newi = (R + G + B) / 3.0;
			if (newi < 0)  newi = 0;
			else if (newi > 1) newi = 1.0;
			if (newi == 0 || mx == mi) {
				s = 0;
				h = 0;
			}
			else {
				s = 1 - mi / newi;
				th = (R - G) * (R - G) + (R - B) * (G - B);
				th = sqrt(th) + 1e-5;
				th = acos(((R - G + R - B)*0.5) / th);
				if (G >= B) h = th;
				else h = 2 * PI - th;
			}
			h = h / (2 * PI);
			H.at<double>(i, j) = h;
			S.at<double>(i, j) = s;
			I.at<double>(i, j) = newi;

			dsthsi.at<Vec3d>(i, j)[2] = h;
			dsthsi.at<Vec3d>(i, j)[1] = s;
			dsthsi.at<Vec3d>(i, j)[0] = newi;
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
			if (I.at<double>(i, j) <= ThresHold) {
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
			if (I.at<double>(i, j) <= ThresHold) {
				D = A;
				C = 1.0 / log2(D + 1);
			}
			else {
				D = (double)(ThresHold * A - ThresHold) / double((1 - ThresHold) * (I.at<double>(i, j))) - (double)(ThresHold * A - 1) / (1 - ThresHold);
				C = 1.0 / log2(D + 1);
			}
			I.at<double>(i, j) = (C * log2(D * (double)I.at<double>(i, j) + 1));
		}
	}//

	 //HSI2RGB
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			double preh = H.at<double>(i, j) * 2 * PI;//?
			double pres = S.at<double>(i, j);
			double prei = I.at<double>(i, j);
			double r = 0, g = 0, b = 0;
			double t1, t2, t3;
			t1 = (1.0 - pres) / 3.0;
			if (preh >= 0 && preh < (PI * 2 / 3)) {
				b = t1;
				t2 = pres * cos(preh);
				t3 = cos(PI / 3 - preh);
				r = (1 + t2 / t3) / 3;
				//g = 1.0 - r - b;
				r = 3 * prei * r;
				//g = 3 * g * prei;
				b = 3 * prei * b;
				g = 3 * prei - (r + b);
			}
			else if (preh >= (PI * 2 / 3) && preh < (PI * 4 / 3)) {
				r = t1;
				t2 = pres * cos(preh - 2 * PI / 3);
				t3 = cos(PI - preh);
				g = (1 + t2 / t3) / 3;
				//b = 1 - r - g;
				r = 3 * prei * r;
				g = 3 * g * prei;
				//b = 3 * prei * b;
				b = 3 * prei - (r + g);
			}
			else if (preh >= (PI * 4 / 3) && preh <= (PI * 2)) {
				g = t1;
				t2 = pres * cos(preh - 4 * PI / 3);
				t3 = cos(PI * 5 / 3 - preh);
				b = (1 + t2 / t3) / 3;
				//r = 1 - g - b;
				//r = 3 * prei * r;
				g = 3 * g * prei;
				b = 3 * prei * b;
				r = 3 * prei - (g + b);
			}

			dst.at<Vec3d>(i, j)[0] = b;
			dst.at<Vec3d>(i, j)[1] = g;
			dst.at<Vec3d>(i, j)[2] = r;

			/*
			dst.at<Vec3b>(i, j)[0] =(b * 255.0);
			dst.at<Vec3b>(i, j)[1] = (int)(g * 255.0);
			dst.at<Vec3b>(i, j)[2] = (int)(r * 255.0);*/
			//printf("%d %d %d\n", (int)(r*255), (int)(g*255), (int)(b*255));	
		}
	}
	//}

	return dst;
	//return dsthsi;
}

int main() {
	
}