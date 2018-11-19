#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace cv;
using namespace std;

int rows, cols;
//获取最小值矩阵
int **getMinChannel(cv::Mat img){
    rows = img.rows;
    cols = img.cols;
    if(img.channels() != 3){
        fprintf(stderr, "Input Error!");
        exit(-1);
    }
    int **imgGray;
    imgGray = new int *[rows];
    for(int i = 0; i < rows; i++){
        imgGray[i] = new int [cols];
    }
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            int loacalMin = 255;
            for(int k = 0; k < 3; k++){
                if(img.at<Vec3b>(i, j)[k] < loacalMin){
                    loacalMin = img.at<Vec3b>(i, j)[k];
                }
            }
            imgGray[i][j] = loacalMin;
        }
    }
    return imgGray;
}

//求暗通道
int **getDarkChannel(int **img, int blockSize = 3){
    if(blockSize%2 == 0 || blockSize < 3){
        fprintf(stderr, "blockSize is not odd or too small!");
        exit(-1);
    }
    //计算pool Size
    int poolSize = (blockSize - 1) / 2;
    int newHeight = rows + poolSize - 1;
    int newWidth = cols + poolSize - 1;
    int **imgMiddle;
    imgMiddle = new int *[newHeight];
    for(int i = 0; i < newHeight; i++){
        imgMiddle[i] = new int [newWidth];
    }
    for(int i = 0; i < newHeight; i++){
        for(int j = 0; j < newWidth; j++){
            if(i < rows && j < cols){
                imgMiddle[i][j] = img[i][j];
            }else{
                imgMiddle[i][j] = 255;
            }
        }
    }
    int **imgDark;
    imgDark = new int *[rows];
    for(int i = 0; i < rows; i++){
        imgDark[i] = new int [cols];
    }
    int localMin = 255;
    for(int i = poolSize; i < newHeight - poolSize; i++){
        for(int j = poolSize; j < newWidth - poolSize; j++){
            for(int k = i-poolSize; k < i+poolSize+1; k++){
                for(int l = j-poolSize; l < j+poolSize+1; l++){
                    if(imgMiddle[k][l] < localMin){
                        localMin = imgMiddle[k][l];
                    }
                }
            }
            imgDark[i-poolSize][j-poolSize] = localMin;
        }
    }
    return imgDark;
}

struct node{
    int x, y, val;
    node(){}
    node(int _x, int _y, int _val):x(_x),y(_y),val(_val){}
    bool operator<(const node &rhs){
        return val > rhs.val;
    }
};

//估算全局大气光值
int getGlobalAtmosphericLightValue(int **darkChannel, cv::Mat img, bool meanMode = false, float percent = 0.001){
    int size = rows * cols;
    std::vector <node> nodes;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            node tmp;
            tmp.x = i, tmp.y = j, tmp.val = darkChannel[i][j];
            nodes.push_back(tmp);
        }
    }
    sort(nodes.begin(), nodes.end());
    int atmosphericLight = 0;
    if(int(percent*size) == 0){
        for(int i = 0; i < 3; i++){
            if(img.at<Vec3b>(nodes[0].x, nodes[0].y)[i] > atmosphericLight){
                atmosphericLight = img.at<Vec3b>(nodes[0].x, nodes[0].y)[i];
            }
        }
    }
    //开启均值模式
    if(meanMode == true){
        int sum = 0;
        for(int i = 0; i < int(percent*size); i++){
            for(int j = 0; j < 3; j++){
                sum = sum + img.at<Vec3b>(nodes[i].x, nodes[i].y)[j];
            }
        }
    }
    //获取暗通道在前0.1%的位置的像素点在原图像中的最高亮度值
    for(int i = 0; i < int(percent*size); i++){
        for(int j = 0; j < 3; j++){
            if(img.at<Vec3b>(nodes[i].x, nodes[i].y)[j] > atmosphericLight){
                atmosphericLight = img.at<Vec3b>(nodes[i].x, nodes[i].y)[j];
            }
        }
    }
    return atmosphericLight;
}

//恢复原图像
// Omega 去雾比例 参数
//t0 最小透射率值
cv::Mat getRecoverScene(cv::Mat img, float omega=0.95, float t0=0.1, int blockSize=15, bool meanModel=false, float percent=0.001){
    int** imgGray = getMinChannel(img);
    int **imgDark = getDarkChannel(imgGray, blockSize=blockSize);
    int atmosphericLight = getGlobalAtmosphericLightValue(imgDark, img, meanModel=meanModel, percent=percent);
    float **imgDark2, **transmission;
    imgDark2 = new float *[rows];
    for(int i = 0; i < rows; i++){
        imgDark2[i] = new float [cols];
    }
    transmission = new float *[rows];
    for(int i = 0; i < rows; i++){
        transmission[i] = new float [cols];
    }
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            imgDark2[i][j] = float(imgDark[i][j]);
            transmission[i][j] = 1 - omega * imgDark[i][j] / atmosphericLight;
            if(transmission[i][j] < 0.1){
                transmission[i][j] = 0.1;
            }
        }
    }
    cv::Mat dst(img.rows, img.cols, CV_8UC3);
    for(int channel = 0; channel < 3; channel++){
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                int temp = (img.at<Vec3b>(i, j)[channel] - atmosphericLight) / transmission[i][j] + atmosphericLight;
                if(temp > 255){
                    temp = 255;
                }
                if(temp < 0){
                    temp = 0;
                }
                dst.at<Vec3b>(i, j)[channel] = temp;
            }
        }
    }
    return dst;
}

int main(){
    cv::Mat src = cv::imread("/home/zxy/CLionProjects/Acmtest/3.jpg");
    rows = src.rows;
    cols = src.cols;
    cv::Mat dst = getRecoverScene(src);
    cv::imshow("origin", src);
    cv::imshow("result", dst);
    cv::imwrite("../zxy.jpg", dst);
    waitKey(0);
}