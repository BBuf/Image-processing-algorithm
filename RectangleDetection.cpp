#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;



//针对灰度图的中值滤波+CVPR 2019的SideWindowFilter
//其他种类的滤波直接换核即可

//记录每一个方向的核的不为0的元素个数
int cnt[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
// 记录每一个方向的滤波器
vector <int> filter[8];

//初始化半径为radius的滤波器，原理可以看https://mp.weixin.qq.com/s/vjzZjRoQw7MnkqAfvwBUNA
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

//实现Side Window Filter的中值滤波，强制保边
Mat MedianSideWindowFilter(Mat src, int radius = 1) {
	int row = src.rows;
	int col = src.cols;
	int channels = src.channels();
	InitFilter(radius);
	//针对灰度图
	vector <int> now;
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
					now.clear();
					for (int x = -radius; x <= radius; x++) {
						for (int y = -radius; y <= radius; y++) {
							//if (x == 0 && y == 0) continue;
							if (filter[k][id]) now.push_back(src.at<uchar>(i + x, j + y) * filter[k][id]);
							id++;
							//val += src.at<uchar>(i + x, j + y) * filter[k][id++];
						}
					}
					sort(now.begin(), now.end());
					int mid = (int)(now.size());
					val = now[mid / 2];
					if (abs(val - src.at<uchar>(i, j)) < minn) {
						minn = abs(val - src.at<uchar>(i, j));
						pos = k;
					}
				}
				int val = 0;
				int id = 0;
				now.clear();
				for (int x = -radius; x <= radius; x++) {
					for (int y = -radius; y <= radius; y++) {
						//if (x == 0 && y == 0) continue;
						if (filter[pos][id]) now.push_back(src.at<uchar>(i + x, j + y) * filter[pos][id]);
						id++;
						//val += src.at<uchar>(i + x, j + y) * filter[k][id++];
					}
				}
				sort(now.begin(), now.end());
				int mid = (int)(now.size());
				val = now[mid / 2];
				dst.at<uchar>(i, j) = val;
			}
		}
		return dst;
	}
	//针对RGB图
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
					now.clear();
					for (int x = -radius; x <= radius; x++) {
						for (int y = -radius; y <= radius; y++) {
							//if (x == 0 && y == 0) continue;
							//val += src.at<Vec3b>(i + x, j + y)[c] * filter[k][id++];
							if (filter[k][id]) now.push_back(src.at<Vec3b>(i + x, j + y)[c] * filter[k][id]);
							id++;
						}
					}
					sort(now.begin(), now.end());
					int mid = (int)(now.size());
					val = now[mid / 2];
					if (abs(val - src.at<Vec3b>(i, j)[c]) < minn) {
						minn = abs(val - src.at<Vec3b>(i, j)[c]);
						pos = k;
					}
				}
				int val = 0;
				int id = 0;
				now.clear();
				for (int x = -radius; x <= radius; x++) {
					for (int y = -radius; y <= radius; y++) {
						//if (x == 0 && y == 0) continue;
						//val += src.at<Vec3b>(i + x, j + y)[c] * filter[k][id++];
						if (filter[pos][id]) now.push_back(src.at<Vec3b>(i + x, j + y)[c] * filter[pos][id]);
						id++;
					}
				}
				sort(now.begin(), now.end());
				int mid = (int)(now.size());
				val = now[mid / 2];
				dst.at<Vec3b>(i, j)[c] = val;
			}
		}
	}
	return dst;
}



const double eps = 1e-7;

//获取pt0->pt1向量和pt0->pt2向量之间的夹角
static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + eps);
}

//寻找矩形
static void findSquares(const Mat& image, vector<vector<Point> >& squares, int N = 5, int thresh = 50)
{

	//滤波可以提升边缘检测的性能
	Mat timg(image);
	// 普通中值滤波
	//medianBlur(image, timg, 9);
	// SideWindowFilter的中值滤波
	timg = MedianSideWindowFilter(image, 4);
	Mat gray0(timg.size(), CV_8U), gray;
	// 存储轮廓
	vector<vector<Point> > contours;

	// 在图像的每一个颜色通道寻找矩形
	for (int c = 0; c < 3; c++)
	{
		int ch[] = { c, 0 };
		// 函数功能：mixChannels主要就是把输入的矩阵（或矩阵数组）的某些通道拆分复制给对应的输出矩阵（或矩阵数组）的某些通道中，其中的对应关系就由fromTo参数制定.
		// 接口：void  mixChannels (const Mat*  src , int  nsrc , Mat*  dst , int  ndst , const int*  fromTo , size_t  npairs );
		// src: 输入矩阵，可以为一个也可以为多个，但是矩阵必须有相同的大小和深度.
		// nsrc: 输入矩阵的个数.
		// dst: 输出矩阵，可以为一个也可以为多个，但是所有的矩阵必须事先分配空间（如用create），大小和深度须与输入矩阵等同.
		// ndst: 输出矩阵的个数
		// fromTo:设置输入矩阵的通道对应输出矩阵的通道，规则如下：首先用数字标记输入矩阵的各个通道。输入矩阵个数可能多于一个并且每个矩阵的通道可能不一样，
		// 第一个输入矩阵的通道标记范围为：0 ~src[0].channels() - 1，第二个输入矩阵的通道标记范围为：src[0].channels() ~src[0].channels() + src[1].channels() - 1,
		// 以此类推；其次输出矩阵也用同样的规则标记，第一个输出矩阵的通道标记范围为：0 ~dst[0].channels() - 1，第二个输入矩阵的通道标记范围为：dst[0].channels()
		// ~dst[0].channels() + dst[1].channels() - 1, 以此类推；最后，数组fromTo的第一个元素即fromTo[0]应该填入输入矩阵的某个通道标记，而fromTo的第二个元素即
		// fromTo[1]应该填入输出矩阵的某个通道标记，这样函数就会把输入矩阵的fromTo[0]通道里面的数据复制给输出矩阵的fromTo[1]通道。fromTo后面的元素也是这个
		// 道理，总之就是一个输入矩阵的通道标记后面必须跟着个输出矩阵的通道标记.
		// npairs: 即参数fromTo中的有几组输入输出通道关系，其实就是参数fromTo的数组元素个数除以2.
		mixChannels(&timg, 1, &gray0, 1, ch, 1);

		// 尝试几个不同的阈值
		for (int l = 0; l < N; l++)
		{
			// hack: use Canny instead of zero threshold level.
			// Canny helps to catch squares with gradient shading
			// 在级别为0的时候不使用阈值为0，而是使用Canny边缘检测算子
			if (l == 0)
			{
				// void Canny(	InputArray image, OutputArray edges, double threshold1, double threshold2, int apertureSize = 3, bool L2gradient = false);
				// 第一个参数：输入图像（八位的图像）
				// 第二个参数：输出的边缘图像
				// 第三个参数：下限阈值，如果像素梯度低于下限阈值，则将像素不被认为边缘
				// 第四个参数：上限阈值，如果像素梯度高于上限阈值，则将像素被认为是边缘（建议上限是下限的2倍或者3倍）
				// 第五个参数：为Sobel()运算提供内核大小，默认值为3
				// 第六个参数：计算图像梯度幅值的标志，默认值为false
				Canny(gray0, gray, 5, thresh, 5);
				// 执行形态学膨胀操作
				dilate(gray, gray, Mat(), Point(-1, -1));
			}
			else
			{
				// 当l不等于0的时候，执行 tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
				gray = gray0 >= (l + 1) * 255 / N;
			}

			// 寻找轮廓并将它们全部存储为列表
			findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

			//存储一个多边形（矩形）
			vector<Point> approx;

			// 测试每一个轮廓
			for (size_t i = 0; i < contours.size(); i++)
			{
				// 近似轮廓，精度与轮廓周长成正比,主要功能是把一个连续光滑曲线折线化，对图像轮廓点进行多边形拟合。
				// 函数声明：void approxPolyDP(InputArray curve, OutputArray approxCurve, double epsilon, bool closed)
				// InputArray curve:一般是由图像的轮廓点组成的点集
				// OutputArray approxCurve：表示输出的多边形点集
				// double epsilon：主要表示输出的精度，就是两个轮廓点之间最大距离数，5,6,7，，8，，,,，
				// bool closed：表示输出的多边形是否封闭

				// arcLength 计算图像轮廓的周长
				approxPolyDP(Mat(contours[i]), approx, 20, true);

				// 近似后，方形轮廓应具有4个顶点
				// 相对较大的区域（以滤除嘈杂的轮廓）并且是凸集。
				// 注意: 使用面积的绝对值，因为面积可以是正值或负值-根据轮廓方向
				if (approx.size() == 4 &&
					fabs(contourArea(Mat(approx))) > 1000 &&
					isContourConvex(Mat(approx)))
				{
					double maxCosine = 0;

					for (int j = 2; j < 5; j++)
					{
						// 找到相邻边之间的角度的最大余弦
						double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
						maxCosine = MAX(maxCosine, cosine);
					}

					// 如果所有角度的余弦都很小(所有角度均为90度)，将顶点集合写入结果vector
					if (maxCosine < 0.3)
						squares.push_back(approx);
				}
			}
		}
	}
}

//在图像上画出方形
void drawSquares(Mat &image, const vector<vector<Point> >& squares) {
	for (size_t i = 0; i < squares.size(); i++)
	{
		const Point* p = &squares[i][0];

		int n = (int)squares[i].size();
		//不检测边界
		if (p->x > 3 && p->y > 3)
			polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
	}
}

int main() {
	Mat src = cv::imread("F:\\stone.jpg");
	vector<vector<Point> > squares;
	findSquares(src, squares, 5, 50);
	drawSquares(src, squares);
	imshow("result", src);
	imwrite("F:\\res2.jpg", src);
	waitKey(0);
	return 0;
}