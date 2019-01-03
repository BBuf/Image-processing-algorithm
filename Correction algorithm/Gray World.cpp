#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
#include "algorithm"
#include "vector"
#include "stdio.h"
#include "map"
#include "unordered_map"
using namespace std;
using namespace cv;

Mat GrayWorld(const Mat &src){
    vector <Mat> bgr;
    cv::split(src, bgr);
    double B = 0;
    double G = 0;
    double R = 0;
    int row = src.rows;
    int col = src.cols;
    Mat dst(row, col, CV_8UC3);
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
              B += 1.0 * src.at<Vec3b>(i, j)[0];
              G += 1.0 * src.at<Vec3b>(i, j)[1];
              R += 1.0 * src.at<Vec3b>(i, j)[2];
        }
    }
    B /= (row * col);
    G /= (row * col);
    R /= (row * col);
    printf("%.5f %.5f %.5f\n", B, G, R);
    double GrayValue = (B + G + R) / 3;
    printf("%.5f\n", GrayValue);
    double kr = GrayValue / R;
    double kg = GrayValue / G;
    double kb = GrayValue / B;
    printf("%.5f %.5f %.5f\n", kb, kg, kr);
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            dst.at<Vec3b>(i, j)[0] = (int)(kb * src.at<Vec3b>(i, j)[0]);
            dst.at<Vec3b>(i, j)[1] = (int)(kg * src.at<Vec3b>(i, j)[1]);
            dst.at<Vec3b>(i, j)[2] = (int)(kr * src.at<Vec3b>(i, j)[2]);
            for(int k = 0; k < 3; k++){
                if(dst.at<Vec3b>(i, j)[k] > 255){
                    dst.at<Vec3b>(i, j)[k] = 255;
                }
            }
        }
    }
    return dst;
}



int main(){
    Mat src = cv::imread("../1.jpg");
    Mat dst = GrayWorld(src);
    cv::imshow("origin", src);
    cv::imshow("result", dst);
    cv::imwrite("../result.jpg", dst);
    cv::waitKey(0);
    return 0;
}