#include "opencv2/opencv.hpp"
#include "iostream"
#include "algorithm"
#include "vector"
#include "stdio.h"
using namespace std;
using namespace cv;

void BoxFilter(const Mat &src, Mat &dst, int ksize){
    CV_Assert(src.channels()==1 || src.channels() == 3); //只处理单通道或者三通道图像
    //生成一维的
    double *matrix = new double[ksize];
    double sum = 0;
    int origin = ksize / 2;
    for(int i = 0; i < ksize; i++){
        double g =1.0;
        sum += g;
        matrix[i] = g;
    }
    for(int i = 0; i < ksize; i++) matrix[i] /= sum;
    int border = ksize / 2;
    copyMakeBorder(src, dst, border, border, border, border, BORDER_CONSTANT);
    int channels = dst.channels();
    int rows = dst.rows - border;
    int cols = dst.cols - border;
    //水平方向
    for(int i = border; i < rows; i++){
        for(int j = border; j < cols; j++){
            double sum[3] = {0};
            for(int k = -border; k<=border; k++){
                if(channels == 1){
                    sum[0] += matrix[border + k] * dst.at<uchar>(i, j+k);
                }else if(channels == 3){
                    Vec3b rgb = dst.at<Vec3b>(i, j+k);
                    sum[0] += matrix[border+k] * rgb[0];
                    sum[1] += matrix[border+k] * rgb[1];
                    sum[2] += matrix[border+k] * rgb[2];
                }
            }
            for(int k = 0; k < channels; k++){
                if(sum[k] < 0) sum[k] = 0;
                else if(sum[k] > 255) sum[k] = 255;
            }
            if(channels == 1)
                dst.at<Vec3b>(i, j) = static_cast<uchar>(sum[0]);
            else if(channels == 3){
                Vec3b rgb = {static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2])};
                dst.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    //竖直方向
    for(int i = border; i < rows; i++){
        for(int j = border; j < cols; j++){
            double sum[3] = {0};
            for(int k = -border; k<=border; k++){
                if(channels == 1){
                    sum[0] += matrix[border + k] * dst.at<uchar>(i+k, j);
                }else if(channels == 3){
                    Vec3b rgb = dst.at<Vec3b>(i+k, j);
                    sum[0] += matrix[border+k] * rgb[0];
                    sum[1] += matrix[border+k] * rgb[1];
                    sum[2] += matrix[border+k] * rgb[2];
                }
            }
            for(int k = 0; k < channels; k++){
                if(sum[k] < 0) sum[k] = 0;
                else if(sum[k] > 255) sum[k] = 255;
            }
            if(channels == 1)
                dst.at<Vec3b>(i, j) = static_cast<uchar>(sum[0]);
            else if(channels == 3){
                Vec3b rgb = {static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2])};
                dst.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    delete [] matrix;
}

int main(){
    Mat src = imread("../beauty.jpg");
    Mat dst(src.rows, src.cols, CV_8UC3);
    BoxFilter(src, dst, 5);
    imshow("origin", src);
    imshow("origin", dst);
    waitKey(0);
    imwrite("../result.jpg", dst);
    return 0;
}