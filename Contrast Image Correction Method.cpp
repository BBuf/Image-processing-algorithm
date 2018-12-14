#include "opencv2/opencv.hpp"
#include "iostream"
#include "algorithm"
#include "vector"
#include "stdio.h"
using namespace std;
using namespace cv;

Mat ContrastImageCorrection(Mat src){
    int rows = src.rows;
    int cols = src.cols;
    Mat yuvImg;
    cvtColor(src, yuvImg, CV_BGR2YUV_I420);
    vector <Mat> mv;
    split(yuvImg, mv);
    Mat OldY = mv[0].clone();
//    for(int i = 0; i < rows; i++){
//        for(int j = 0; j < cols; j++){
//            mv[0].at<uchar>(i, j) = 255 - mv[0].at<uchar>(i, j);
//        }
//    }
    Mat temp;
    bilateralFilter(mv[0], temp, 9, 50, 50);
    //GaussianBlur(mv[0], temp, Size(41, 41), BORDER_DEFAULT);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            float Exp = pow(2, (128 - (255 - temp.at<uchar>(i, j))) / 128.0);
            int value = int(255 * pow(OldY.at<uchar>(i, j) / 255.0, Exp));
            temp.at<uchar>(i, j) = value;
        }
    }
    Mat dst(rows, cols, CV_8UC3);
//    mv[0] = temp;
//    merge(mv, dst);
//    cvtColor(dst, dst, CV_YUV2BGRA_I420);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++) {
            if (OldY.at<uchar>(i, j) == 0) {
                for (int k = 0; k < 3; k++) dst.at<Vec3b>(i, j)[k] = 0;
            } else {
                //channel B
                dst.at<Vec3b>(i, j)[0] =
                        (temp.at<uchar>(i, j)  * (src.at<Vec3b>(i, j)[0] + OldY.at<uchar>(i, j)) / OldY.at<uchar>(i, j) +
                         src.at<Vec3b>(i, j)[0] - OldY.at<uchar>(i, j)) >> 1;
                //channel G
                dst.at<Vec3b>(i, j)[1] =
                        (temp.at<uchar>(i, j)  * (src.at<Vec3b>(i, j)[1] + OldY.at<uchar>(i, j)) / OldY.at<uchar>(i, j) +
                         src.at<Vec3b>(i, j)[1] - OldY.at<uchar>(i, j)) >> 1;
                //channel R
                dst.at<Vec3b>(i, j)[2] =
                        (temp.at<uchar>(i, j) * (src.at<Vec3b>(i, j)[2] + OldY.at<uchar>(i, j)) / OldY.at<uchar>(i, j) +
                         src.at<Vec3b>(i, j)[2] - OldY.at<uchar>(i, j)) >> 1;
            }
        }
    }
//    for(int i = 0; i < rows; i++){
//        for(int j = 0; j < cols; j++){
//            for(int k = 0; k < 3; k++){
//                if(dst.at<Vec3b>(i, j)[k] < 0){
//                    dst.at<Vec3b>(i, j)[k] = 0;
//                }else if(dst.at<Vec3b>(i, j)[k] > 255){
//                    dst.at<Vec3b>(i, j)[k] = 255;
//                }
//            }
//        }
//    }
    return dst;
}

int main(){
    Mat src = imread("/home/streamax/CLionProjects/Paper/1.jpg");
    Rect rect(0, 0, (src.cols-1)/2*2, (src.rows-1)/2*2);
    Mat newsrc = src(rect);
    Mat dst = ContrastImageCorrection(newsrc);
    imshow("origin", newsrc);
    imshow("result", dst);
    cv::imwrite("../result.jpg", dst);
    waitKey(0);
    return 0;
}