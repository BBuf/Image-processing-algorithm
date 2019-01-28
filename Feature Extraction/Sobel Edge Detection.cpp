#include "opencv2/opencv.hpp"
#include "iostream"
#include "algorithm"
#include "vector"
#include "stdio.h"
using namespace std;
using namespace cv;

const int fac[9]={1, 1, 2, 6, 24, 120, 720, 5040, 40320};
//Sobel平滑算子
Mat getSmmoothKernel(int ksize){
    Mat Smooth = Mat::zeros(Size(ksize, 1), CV_32FC1);
    for(int i = 0; i < ksize; i++){
        Smooth.at<float>(0, i) = float(fac[ksize-1]/(fac[i] * fac[ksize-1-i]));
    }
    return Smooth;
}
//Sobel差分算子
Mat getDiffKernel(int ksize){
    Mat Diff = Mat::zeros(Size(ksize, 1), CV_32FC1);
    Mat preDiff = getSmmoothKernel(ksize-1);
    for(int i = 0; i < ksize; i++){
        if(i == 0){
            Diff.at<float>(0, i) = 1;
        }else if(i == ksize-1){
            Diff.at<float>(0, i) = -1;
        }else{
            Diff.at<float>(0, i) = preDiff.at<float>(0, i) - preDiff.at<float>(0, i-1);
        }
    }
    return Diff;
}
//调用filter2D实现卷积
void conv2D(InputArray src, InputArray kernel, OutputArray dst, int dep, Point anchor=Point(-1,-1), int borderType=BORDER_DEFAULT){
    Mat kernelFlip;
    flip(kernel, kernelFlip, -1);
    filter2D(src, dst, dep, kernelFlip, anchor, 0.0, borderType);
}
//先进行垂直方向的卷积，再进行水平方向的卷积
void sepConv2D_Y_X(InputArray src, OutputArray dst, int dep, InputArray kernelY, InputArray kernelX, Point anchor=Point(-1,-1), int borderType=BORDER_DEFAULT){
    Mat Y;
    conv2D(src, kernelY, Y, dep, anchor, borderType);
    conv2D(Y, kernelX, dst, dep, anchor, borderType);
}
//先进行水平方向的卷积，再进行垂直方向的卷积
void sepConv2D_X_Y(InputArray src, OutputArray dst, int dep, InputArray kernelX, InputArray kernelY, Point anchor=Point(-1,-1), int borderType=BORDER_DEFAULT){
    Mat X;
    conv2D(src, kernelX, X, dep, anchor, borderType);
    conv2D(X, kernelY, dst, dep, anchor, borderType);
}
//Sobel算子提取边缘信息
Mat Sobel(Mat &src, int x_flag, int y_flag, int kSize, int borderType){
    Mat Smooth = getSmmoothKernel(kSize);
    Mat Diff = getDiffKernel(kSize);
    Mat dst;
    if(x_flag){
        sepConv2D_Y_X(src, dst, CV_32FC1, Smooth.t(), Diff, Point(-1, -1), borderType);
    }else if(x_flag == 0 && y_flag){
        sepConv2D_X_Y(src, dst, CV_32FC1, Smooth, Diff.t(), Point(-1, -1), borderType);
    }
    return dst;
}
int main(){
    Mat src = imread("../lena.jpg");
    Mat gray;
    cvtColor(src, gray, CV_BGR2GRAY);
    Mat dst1 = Sobel(gray, 1, 0, 3, BORDER_DEFAULT);
    Mat dst2 = Sobel(gray, 0, 1, 3, BORDER_DEFAULT);
    //转8位灰度图显示
    convertScaleAbs(dst1, dst1);
    convertScaleAbs(dst2, dst2);
    imshow("origin", gray);
    imshow("result-X", dst1);
    imshow("result-Y", dst2);
    imwrite("../result.jpg", dst1);
    imwrite("../result2.jpg", dst2);
    waitKey(0);
    return 0;
}