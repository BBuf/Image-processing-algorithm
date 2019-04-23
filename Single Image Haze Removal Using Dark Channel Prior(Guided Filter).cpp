#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace cv;
using namespace std;

cv::Mat guidedFilter(cv::Mat I, cv::Mat p, int r, double eps)
{
	/*
	% GUIDEDFILTER   O(1) time implementation of guided filter.
	%
	%   - guidance image: I (should be a gray-scale/single channel image)
	%   - filtering input image: p (should be a gray-scale/single channel image)
	%   - local window radius: r
	%   - regularization parameter: eps
	*/

	cv::Mat _I;
	I.convertTo(_I, CV_64FC1);
	I = _I;

	cv::Mat _p;
	p.convertTo(_p, CV_64FC1);
	p = _p;

	//[hei, wid] = size(I);
	int hei = I.rows;
	int wid = I.cols;

	//N = boxfilter(ones(hei, wid), r); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.
	cv::Mat N;
	cv::boxFilter(cv::Mat::ones(hei, wid, I.type()), N, CV_64FC1, cv::Size(r, r));

	//mean_I = boxfilter(I, r) ./ N;
	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, CV_64FC1, cv::Size(r, r));

	//mean_p = boxfilter(p, r) ./ N;
	cv::Mat mean_p;
	cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));

	//mean_Ip = boxfilter(I.*p, r) ./ N;
	cv::Mat mean_Ip;
	cv::boxFilter(I.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));

	//cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//mean_II = boxfilter(I.*I, r) ./ N;
	cv::Mat mean_II;
	cv::boxFilter(I.mul(I), mean_II, CV_64FC1, cv::Size(r, r));

	//var_I = mean_II - mean_I .* mean_I;
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);

	//a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;	
	cv::Mat a = cov_Ip / (var_I + eps);

	//b = mean_p - a .* mean_I; % Eqn. (6) in the paper;
	cv::Mat b = mean_p - a.mul(mean_I);

	//mean_a = boxfilter(a, r) ./ N;
	cv::Mat mean_a;
	cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r));
	mean_a = mean_a / N;

	//mean_b = boxfilter(b, r) ./ N;
	cv::Mat mean_b;
	cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));
	mean_b = mean_b / N;

	//q = mean_a .* I + mean_b; % Eqn. (8) in the paper;
	cv::Mat q = mean_a.mul(I) + mean_b;

	return q;
}


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

struct node {
	int x, y, val;
	node() {}
	node(int _x, int _y, int _val) :x(_x), y(_y), val(_val) {}
	bool operator<(const node &rhs) {
		return val > rhs.val;
	}
};

//估算全局大气光值
int getGlobalAtmosphericLightValue(int **darkChannel, cv::Mat img, bool meanMode = false, float percent = 0.001) {
	int size = rows * cols;
	std::vector <node> nodes;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			node tmp;
			tmp.x = i, tmp.y = j, tmp.val = darkChannel[i][j];
			nodes.push_back(tmp);
		}
	}
	sort(nodes.begin(), nodes.end());
	int atmosphericLight = 0;
	if (int(percent*size) == 0) {
		for (int i = 0; i < 3; i++) {
			if (img.at<Vec3b>(nodes[0].x, nodes[0].y)[i] > atmosphericLight) {
				atmosphericLight = img.at<Vec3b>(nodes[0].x, nodes[0].y)[i];
			}
		}
	}
	//开启均值模式
	if (meanMode == true) {
		int sum = 0;
		for (int i = 0; i < int(percent*size); i++) {
			for (int j = 0; j < 3; j++) {
				sum = sum + img.at<Vec3b>(nodes[i].x, nodes[i].y)[j];
			}
		}
	}
	//获取暗通道在前0.1%的位置的像素点在原图像中的最高亮度值
	for (int i = 0; i < int(percent*size); i++) {
		for (int j = 0; j < 3; j++) {
			if (img.at<Vec3b>(nodes[i].x, nodes[i].y)[j] > atmosphericLight) {
				atmosphericLight = img.at<Vec3b>(nodes[i].x, nodes[i].y)[j];
			}
		}
	}
	return atmosphericLight;
}

//恢复原图像
// Omega 去雾比例 参数
//t0 最小透射率值
cv::Mat getRecoverScene(cv::Mat img, float omega = 0.95, float t0 = 0.1, int blockSize = 15, bool meanModel = false, float percent = 0.001) {
	int** imgGray = getMinChannel(img);
	int **imgDark = getDarkChannel(imgGray, blockSize = blockSize);
	int atmosphericLight = getGlobalAtmosphericLightValue(imgDark, img, meanModel = meanModel, percent = percent);
	float **imgDark2, **transmission;
	imgDark2 = new float *[rows];
	for (int i = 0; i < rows; i++) {
		imgDark2[i] = new float[cols];
	}
	Mat B(rows, cols, CV_8UC1);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			B.at<uchar>(i, j) = img.at<Vec3b>(i, j)[0];
		}
	}
	Mat p(rows, cols, CV_64FC1);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			p.at<double>(i, j) = 1 - omega * imgDark[i][j] / atmosphericLight;
		}
	}
	Mat transmission_filter = guidedFilter(B, p, 80, 1e-3);
	imshow("transmission", transmission_filter);
	imwrite("F:\\res3.jpg", transmission_filter);
	waitKey(0);
	transmission = new float *[rows];
	for (int i = 0; i < rows; i++) {
		transmission[i] = new float[cols];
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			imgDark2[i][j] = float(imgDark[i][j]);
			//transmission[i][j] = 1 - omega * imgDark[i][j] / atmosphericLight;
			transmission[i][j] = transmission_filter.at<double>(i, j);
			if (transmission[i][j] < 0.1) {
				transmission[i][j] = 0.1;
			}
		}
	}
	cv::Mat dst(img.rows, img.cols, CV_8UC3);
	for (int channel = 0; channel < 3; channel++) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				int temp = (img.at<Vec3b>(i, j)[channel] - atmosphericLight) / transmission[i][j] + atmosphericLight;
				if (temp > 255) {
					temp = 255;
				}
				if (temp < 0) {
					temp = 0;
				}
				dst.at<Vec3b>(i, j)[channel] = temp;
			}
		}
	}
	return dst;
}

int main() {
	Mat src = imread("F:\\fog\\3.jpg");
	//cv::Mat src = cv::imread("/home/zxy/CLionProjects/Acmtest/3.jpg");
	rows = src.rows;
	cols = src.cols;
	cv::Mat dst = getRecoverScene(src);
	cv::imshow("origin", src);
	cv::imshow("result", dst);
	cv::imwrite("F:\\res2.jpg", dst);
	waitKey(0);
}