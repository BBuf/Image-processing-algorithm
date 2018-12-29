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

Mat RGB2GRAY(const Mat &src){
    if(!src.data || src.channels()!=3){
        exit(1);
    }
    int row = src.rows;
    int col = src.cols;
    Mat gray = Mat(row, col, CV_8UC1);
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            gray.at<uchar>(i, j) = (uchar)(0.114 * src.at<Vec3b>(i, j)[0] + 0.587 * src.at<Vec3b>(i, j)[1] + 0.299 * src.at<Vec3b>(i, j)[2]);
        }
    }
    return gray;
}

void SobelGradDirction(Mat &src, Mat &sobelX, Mat &sobelY){
    int row = src.rows;
    int col = src.cols;
    sobelX = Mat::zeros(src.size(), CV_32SC1);
    sobelY = Mat::zeros(src.size(), CV_32SC1);

    for(int i = 0; i < row-1; i++){
        for(int j = 0; j < col-1; j++){
            double gradY = src.at<uchar>(i+1, j-1) + src.at<uchar>(i+1, j) * 2 + src.at<uchar>(i+1, j+1) - src.at<uchar>(i-1, j-1) - src.at<uchar>(i-1, j) * 2 - src.at<uchar>(i-1, j+1);
            sobelY.at<float>(i, j) = abs(gradY);
            double gradX = src.at<uchar>(i-1, j+1) + src.at<uchar>(i, j+1) * 2 + src.at<uchar>(i+1, j+1) - src.at<uchar>(i-1, j-1) - src.at<uchar>(i, j-1) * 2 - src.at<uchar>(i+1, j-1);
            sobelX.at<float>(i, j) = abs(gradX);
        }
    }
    //将梯度数组转换为8位无符号整形
    convertScaleAbs(sobelX, sobelX);
    convertScaleAbs(sobelY, sobelY);
}

Mat SobelXX(const Mat src){
    int row = src.rows;
    int col = src.cols;
    Mat_<float> dst(row, col, CV_32FC1);
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            dst.at<float>(i, j) = src.at<uchar>(i, j) * src.at<uchar>(i, j);
        }
    }
    return dst;
}

Mat SobelYY(const Mat src){
    int row = src.rows;
    int col = src.cols;
    Mat_<float> dst(row, col, CV_32FC1);
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            dst.at<float>(i, j) = src.at<uchar>(i, j) * src.at<uchar>(i, j);
        }
    }
    return dst;
}

Mat SobelXY(const Mat src1, const Mat &src2){
    int row = src1.rows;
    int col = src1.cols;
    Mat_<float> dst(row, col, CV_32FC1);
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            dst.at<float>(i, j) = src1.at<uchar>(i, j) * src2.at<uchar>(i, j);
        }
    }
    return dst;
}

void separateGaussianFilter(Mat_<float> &src, Mat_<float> &dst, int ksize, double sigma){
    CV_Assert(src.channels()==1 || src.channels() == 3); //只处理单通道或者三通道图像
    //生成一维的
    double *matrix = new double[ksize];
    double sum = 0;
    int origin = ksize / 2;
    for(int i = 0; i < ksize; i++){
        double g = exp(-(i-origin) * (i-origin) / (2 * sigma * sigma));
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
    for(int i = border; i < rows - border; i++){
        for(int j = border; j < cols - border; j++){
            float sum[3] = {0};
            for(int k = -border; k<=border; k++){
                if(channels == 1){
                    sum[0] += matrix[border + k] * dst.at<float>(i, j+k);
                }else if(channels == 3){
                    Vec3f rgb = dst.at<Vec3f>(i, j+k);
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
                dst.at<float>(i, j) = static_cast<float>(sum[0]);
            else if(channels == 3){
                Vec3f rgb = {static_cast<float>(sum[0]), static_cast<float>(sum[1]), static_cast<float>(sum[2])};
                dst.at<Vec3f>(i, j) = rgb;
            }
        }
    }
    //竖直方向
    for(int i = border; i < rows - border; i++){
        for(int j = border; j < cols - border; j++){
            float sum[3] = {0};
            for(int k = -border; k<=border; k++){
                if(channels == 1){
                    sum[0] += matrix[border + k] * dst.at<float>(i+k, j);
                }else if(channels == 3){
                    Vec3f rgb = dst.at<Vec3f>(i+k, j);
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
                dst.at<float>(i, j) = static_cast<float>(sum[0]);
            else if(channels == 3){
                Vec3f rgb = {static_cast<float>(sum[0]), static_cast<float>(sum[1]), static_cast<float>(sum[2])};
                dst.at<Vec3f>(i, j) = rgb;
            }
        }
    }
    delete [] matrix;
}

Mat harrisResponse(Mat_<float> GaussXX, Mat_<float> GaussYY, Mat_<float> GaussXY, float k){
    int row = GaussXX.rows;
    int col = GaussXX.cols;
    Mat_<float> dst(row, col, CV_32FC1);
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            float a = GaussXX.at<float>(i, j);
            float b = GaussYY.at<float>(i, j);
            float c = GaussXY.at<float>(i, j);
            dst.at<float>(i, j) = a * b - c * c - k * (a + b) * (a + b);
        }
    }
    return dst;
}

int dir[8][2] = {0, 1, 0, -1, 1, 0, -1, 0, 1, 1, 1, -1, -1, 1, -1, -1};

void MaxLocalValue(Mat_<float>&resultData, Mat srcGray, Mat &resultImage, int kSize){
    int r = kSize / 2;
    resultImage = srcGray.clone();
    int row = resultImage.rows;
    int col = resultImage.cols;
    for(int i = r; i < row - r; i++){
        for(int j = r; j < col - r; j++){
            bool flag = true;
            for(int k = 0; k < 8; k++){
                int tx = i + dir[k][0];
                int ty = j + dir[k][1];
                if(resultData.at<float>(i, j) < resultData.at<float>(tx, ty)){
                    flag = false;
                }
            }
            if(flag && (int)resultData.at<float>(i, j) > 18000){
                circle(resultImage, Point(i, j), 5, Scalar(0, 0, 255), 2, 8, 0);
            }
        }
    }
}

int main(){
    Mat src = imread("../lena.jpg");
    Mat srcGray = RGB2GRAY(src);
    Mat imageSobelX;
    Mat imageSobelY;
    Mat resultImage;
    Mat_<float> imageSobelXX, imageSobelYY, imageSobelXY;
    Mat_<float> GaussianXX, GaussianXY, GaussianYY, HarrisResponse;
    SobelGradDirction(srcGray, imageSobelX, imageSobelY);
    imageSobelXX = SobelXX(imageSobelX);
    imageSobelYY = SobelYY(imageSobelY);
    imageSobelXY = SobelXY(imageSobelX, imageSobelY);
    separateGaussianFilter(imageSobelXX, GaussianXX, 3, 1.0);
    separateGaussianFilter(imageSobelYY, GaussianYY, 3, 1.0);
    separateGaussianFilter(imageSobelXY, GaussianXY, 3, 1.0);
    HarrisResponse = harrisResponse(GaussianXX, GaussianYY, GaussianXY, 0.05);
    MaxLocalValue(HarrisResponse, srcGray, resultImage, 3);
    imshow("origin", src);
    imshow("gray", srcGray);
    imshow("result", resultImage);
    imwrite("../result.jpg", resultImage);
    waitKey(0);
    return 0;
}