#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
#include "algorithm"
#include "vector"
#include "stdio.h"
using namespace std;
using namespace cv;

const double PI = 3.1415926;

double getSum(Mat src){
    double sum = 0;
    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            sum += (double)src.at<double>(i, j);
        }
    }
    return sum;
}

Mat CannyEdgeDetection(cv::Mat src, int kSize, double hightThres, double lowThres){
//    if(src.channels() == 3){
//        cvtColor(src, src, CV_BGR2GRAY);
//    }
    cv::Rect rect;
    Mat gaussResult;
    int row = src.rows;
    int col = src.cols;
    printf("%d %d\n", row, col);
    Mat filterImg = Mat::zeros(row, col, CV_64FC1);
    src.convertTo(src, CV_64FC1);
    Mat dst = Mat::zeros(row, col, CV_64FC1);
    int gaussCenter = kSize / 2;
    double  sigma = 1;
    Mat guassKernel = Mat::zeros(kSize, kSize, CV_64FC1);
    for(int i = 0; i < kSize; i++){
        for(int j = 0; j < kSize; j++){
            guassKernel.at<double>(i, j) = (1.0 / (2.0 * PI * sigma * sigma)) * (double)exp(-(((double)pow((i - (gaussCenter+1)), 2) + (double)pow((j-(gaussCenter+1)), 2)) / (2.0*sigma*sigma)));
        }
    }
    Scalar sumValueScalar = cv::sum(guassKernel);
    double sum = sumValueScalar.val[0];
    guassKernel = guassKernel / sum;
    for(int i = gaussCenter; i < row - gaussCenter; i++){
        for(int j = gaussCenter; j < col - gaussCenter; j++){
            rect.x = j - gaussCenter;
            rect.y = i - gaussCenter;
            rect.width = kSize;
            rect.height = kSize;
            //printf("%d %d\n", i, j);
            //printf("%d %d %.5f\n", i, j, cv::sum(guassKernel.mul(src(rect))).val[0]);
            filterImg.at<double>(i, j) = cv::sum(guassKernel.mul(src(rect))).val[0];
        }
    }
    Mat gradX = Mat::zeros(row, col, CV_64FC1); //水平梯度
    Mat gradY = Mat::zeros(row, col, CV_64FC1); //垂直梯度
    Mat grad = Mat::zeros(row, col, CV_64FC1); //梯度幅值
    Mat thead = Mat::zeros(row, col, CV_64FC1); //梯度角度
    Mat whereGrad = Mat::zeros(row, col, CV_64FC1);//区域
    //x方向的Sobel算子
    Mat Sx = (cv::Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    //y方向的Sobel算子
    Mat Sy = (cv::Mat_<double >(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    //计算梯度赋值和角度
    for(int i=1; i < row-1; i++){
        for(int j=1; j < col-1; j++){
            rect.x = j-1;
            rect.y = i-1;
            rect.width = 3;
            rect.height = 3;
            Mat rectImg = Mat::zeros(3, 3, CV_64FC1);
            filterImg(rect).copyTo(rectImg);
            //梯度和角度
            gradX.at<double>(i, j) += cv::sum(rectImg.mul(Sx)).val[0];
            gradY.at<double>(i, j) += cv::sum(rectImg.mul(Sy)).val[0];
            grad.at<double>(i, j) = sqrt(pow(gradX.at<double>(i, j), 2) + pow(gradY.at<double>(i, j), 2));
            thead.at<double>(i, j) = atan(gradY.at<double>(i, j)/gradX.at<double>(i, j));
            if(0 <= thead.at<double>(i, j) <= (PI/4.0)){
                whereGrad.at<double>(i, j) = 0;
            }else if(PI/4.0 < thead.at<double>(i, j) <= (PI/2.0)){
                whereGrad.at<double>(i, j) = 1;
            }else if(-PI/2.0 <= thead.at<double>(i, j) <= (-PI/4.0)){
                whereGrad.at<double>(i, j) = 2;
            }else if(-PI/4.0 < thead.at<double>(i, j) < 0){
                whereGrad.at<double>(i, j) = 3;
            }
        }
    }
    //printf("success\n");
    //梯度归一化
    double gradMax;
    cv::minMaxLoc(grad, &gradMax);
    if(gradMax != 0){
        grad = grad / gradMax;
    }
    //双阈值确定
    cv::Mat caculateValue = cv::Mat::zeros(row, col, CV_64FC1); //grad变成一维
    resize(grad, caculateValue, Size(1, grad.rows * grad.cols));
    cv::sort(caculateValue, caculateValue, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);//升序
    long long highIndex= row * col * hightThres;
    double highValue = caculateValue.at<double>(highIndex, 0); //最大阈值
    double lowValue = highValue * lowThres;
    //NMS
    for(int i = 1 ; i < row-1; i++ ){
        for( int j = 1; j<col-1; j++){
            // 八个方位
            double N = grad.at<double>(i-1, j);
            double NE = grad.at<double>(i-1, j+1);
            double E = grad.at<double>(i, j+1);
            double SE = grad.at<double>(i+1, j+1);
            double S = grad.at<double>(i+1, j);
            double SW = grad.at<double>(i-1, j-1);
            double W = grad.at<double>(i, j-1);
            double NW = grad.at<double>(i -1, j -1); // 区域判断，线性插值处理
            double tanThead; // tan角度
            double Gp1; // 两个方向的梯度强度
            double Gp2; // 求角度，绝对值
            tanThead = abs(tan(thead.at<double>(i,j)));
            switch ((int)whereGrad.at<double>(i,j)) {
                case 0: Gp1 = (1- tanThead) * E + tanThead * NE; Gp2 = (1- tanThead) * W + tanThead * SW; break;
                case 1: Gp1 = (1- tanThead) * N + tanThead * NE; Gp2 = (1- tanThead) * S + tanThead * SW; break;
                case 2: Gp1 = (1- tanThead) * N + tanThead * NW; Gp2 = (1- tanThead) * S + tanThead * SE; break;
                case 3: Gp1 = (1- tanThead) * W + tanThead *NW; Gp2 = (1- tanThead) * E + tanThead *SE; break;
                default: break;
            }
            // NMS -非极大值抑制和双阈值检测
            if(grad.at<double>(i, j) >= Gp1 && grad.at<double>(i, j) >= Gp2){
                //双阈值检测
                if(grad.at<double>(i, j) >= highValue){
                    grad.at<double>(i, j) = highValue;
                    dst.at<double>(i, j) = 255;
                } else if(grad.at<double>(i, j) < lowValue){
                    grad.at<double>(i, j) = 0;
                } else{
                    grad.at<double>(i, j) = lowValue;
                }
            } else{
                grad.at<double>(i, j) = 0;
            }
        }
    }
    //抑制孤立低阈值点 3*3. 找到高阈值就255
    for(int i = 1; i < row - 1; i++){
        for(int j = 1; j < col - 1; j++){
            if(grad.at<double>(i, j) == lowValue){
                //3*3 区域强度
                rect.x = i-1;
                rect.y = j-1;
                rect.width = 3;
                rect.height = 3;
                for(int x = 0; x < 3; x++){
                    for(int y = 0; y < 3; y++){
                        if(grad(rect).at<double>(x, y)==highValue){
                            dst.at<double>(i, j) = 255;
                            break;
                        }
                    }
                }
            }
        }
    }
    return dst;
}

int main(){
    Mat src = imread("../lena.jpg");
    Mat gray;
    cvtColor(src, gray, CV_BGR2GRAY);
    Mat dst = CannyEdgeDetection(gray, 3, 0.8, 0.5);
    imshow("origin", src);
    imshow("gray", gray);
    imshow("result", dst);
    waitKey(0);
    imwrite("../result.jpg", gray);
    imwrite("../result2.jpg", dst);
    return 0;
}