#include "opencv2/opencv.hpp"
#include "iostream"
#include "algorithm"
#include "vector"
#include "stdio.h"
using namespace std;
using namespace cv;

//直方图均衡化
Mat Histogramequalization(Mat src) {
    int R[256] = {0};
    int G[256] = {0};
    int B[256] = {0};
    int rows = src.rows;
    int cols = src.cols;
    int sum = rows * cols;
    //统计直方图的RGB分布
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            B[src.at<Vec3b>(i, j)[0]]++;
            G[src.at<Vec3b>(i, j)[1]]++;
            R[src.at<Vec3b>(i, j)[2]]++;
        }
    }
    //构建直方图的累计分布方程，用于直方图均衡化
    double val[3] = {0};
    for (int i = 0; i < 256; i++) {
        val[0] += B[i];
        val[1] += G[i];
        val[2] += R[i];
        B[i] = val[0] * 255 / sum;
        G[i] = val[1] * 255 / sum;
        R[i] = val[2] * 255 / sum;
    }
    //归一化直方图
    Mat dst(rows, cols, CV_8UC3);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            dst.at<Vec3b>(i, j)[0] = B[src.at<Vec3b>(i, j)[0]];
            dst.at<Vec3b>(i, j)[1] = B[src.at<Vec3b>(i, j)[1]];
            dst.at<Vec3b>(i, j)[2] = B[src.at<Vec3b>(i, j)[2]];
        }
    }
    return dst;
}

int main(){
    Mat src = imread("../1.jpg");
    Mat dst = Histogramequalization(src);
    imshow("origin", src);
    imshow("result", dst);
    imwrite("../result.jpg", dst);
    waitKey(0);
    return 0;
}