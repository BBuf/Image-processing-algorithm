#include "opencv2/opencv.hpp"
#include "iostream"
#include "algorithm"
#include "vector"
using namespace std;
using namespace cv;

Mat AutoLevel(Mat src, double LowCut, double HighCut){
    int rows = src.rows;
    int cols = src.cols;
    int totalPixel = rows * cols;
    //统计每个通道的直方图
    uchar Pixel[256*3] = {0};
    vector <Mat> rgb;
    split(src, rgb);
    Mat HistBlue, HistGreen, HistRed;
    int histSize = 256;
    float range[] = {0, 255};
    const float* histRange = {range};
    bool uniform = true;
    bool accumulate = false;
    calcHist(&rgb[0], 1, 0, Mat(), HistRed, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&rgb[1], 1, 0, Mat(), HistGreen, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&rgb[2], 1, 0, Mat(), HistBlue, 1, &histSize, &histRange, uniform, accumulate);
    //分别计算各通道按照给定的参数所确定的上下限值
    int MinBlue = 0, MaxBlue = 0;
    int MinRed = 0, MaxRed = 0;
    int MinGreen = 0, MaxGreen = 0;

    //Blue Channel
    float sum = 0;
    sum = 0;
    for(int i = 0; i < 256; i++){
        sum += HistBlue.at<float>(i);
        if(sum >= totalPixel * LowCut * 0.01){
            MinBlue = i;
            break;
        }
    }
    sum = 0;
    for(int i = 255; i >= 0; i--){
        sum = sum + HistBlue.at<float>(i);
        if(sum >= totalPixel * HighCut * 0.01){
            MaxBlue = i;
            break;
        }
    }
    //Red channel
    for(int i = 0; i < 256; i++){
        sum += HistRed.at<float>(i);
        if(sum >= totalPixel * LowCut * 0.01){
            MinRed = i;
            break;
        }
    }
    sum = 0;
    for(int i = 255; i >= 0; i--){
        sum = sum + HistRed.at<float>(i);
        if(sum >= totalPixel * HighCut * 0.01){
            MaxRed = i;
            break;
        }
    }
    //Green channel
    sum = 0;
    for(int i = 0; i < 256; i++){
        sum += HistGreen.at<float>(i);
        if(sum >= totalPixel * LowCut * 0.01){
            MinGreen = i;
            break;
        }
    }
    sum = 0;
    for(int i = 255; i >= 0; i--){
        sum = sum + HistGreen.at<float>(i);
        if(sum >= totalPixel * HighCut * 0.01){
            MaxGreen = i;
            break;
        }
    }
    printf("%d %d %d %d %d %d\n", MinGreen, MaxGreen, MinBlue, MaxBlue, MinRed, MaxRed);
    //自动色阶：按照我们刚刚计算出的MinBlue/MaxBlue构建一个隐射表
    for(int i = 0; i < 256; i++){
        if(i <= MinBlue){
            Pixel[i*3+2] = 0;
        }else{
            if(i > MaxBlue){
                Pixel[i*3+2] = 255;
            }else{
                float temp = (float)(i - MinBlue) / (MaxBlue - MinBlue);
                Pixel[i*3+2] = (uchar)(temp*255);
            }
        }
        if(i <= MinGreen){
            Pixel[i*3+1] = 0;
        }else{
            if(i > MaxGreen){
                Pixel[i*3+1] = 255;
            }else{
                float temp = (float)(i - MinGreen) / (MaxGreen - MinGreen);
                Pixel[i*3+1] = (uchar)(temp*255);
            }
        }
        if(i <= MinRed){
            Pixel[i*3] = 0;
        }else{
            if(i > MaxRed){
                Pixel[i*3] = 255;
            }else{
                float temp = (float)(i - MinRed) / (MaxRed - MinRed);
                Pixel[i*3] = (uchar)(temp*255);
            }
        }
    }
    Mat dst;
    Mat TMP(1, 256, CV_8UC3, Pixel);
    LUT(src, TMP, dst);
    return dst;
}

Mat AutoContrast(Mat src, double LowCut, double HighCut){
    int rows = src.rows;
    int cols = src.cols;
    int totalPixel = rows * cols;
    //统计每个通道的直方图
    uchar Pixel[256*3] = {0};
    vector <Mat> rgb;
    split(src, rgb);
    Mat HistBlue, HistGreen, HistRed;
    int histSize = 256;
    float range[] = {0, 255};
    const float* histRange = {range};
    bool uniform = true;
    bool accumulate = false;
    calcHist(&rgb[0], 1, 0, Mat(), HistRed, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&rgb[1], 1, 0, Mat(), HistGreen, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&rgb[2], 1, 0, Mat(), HistBlue, 1, &histSize, &histRange, uniform, accumulate);
    //分别计算各通道按照给定的参数所确定的上下限值
    int MinBlue = 0, MaxBlue = 0;
    int MinRed = 0, MaxRed = 0;
    int MinGreen = 0, MaxGreen = 0;

    //Blue Channel
    float sum = 0;
    sum = 0;
    for(int i = 0; i < 256; i++){
        sum += HistBlue.at<float>(i);
        if(sum >= totalPixel * LowCut * 0.01){
            MinBlue = i;
            break;
        }
    }
    sum = 0;
    for(int i = 255; i >= 0; i--){
        sum = sum + HistBlue.at<float>(i);
        if(sum >= totalPixel * HighCut * 0.01){
            MaxBlue = i;
            break;
        }
    }
    //Red channel
    for(int i = 0; i < 256; i++){
        sum += HistRed.at<float>(i);
        if(sum >= totalPixel * LowCut * 0.01){
            MinRed = i;
            break;
        }
    }
    sum = 0;
    for(int i = 255; i >= 0; i--){
        sum = sum + HistRed.at<float>(i);
        if(sum >= totalPixel * HighCut * 0.01){
            MaxRed = i;
            break;
        }
    }
    //Green channel
    sum = 0;
    for(int i = 0; i < 256; i++){
        sum += HistGreen.at<float>(i);
        if(sum >= totalPixel * LowCut * 0.01){
            MinGreen = i;
            break;
        }
    }
    sum = 0;
    for(int i = 255; i >= 0; i--){
        sum = sum + HistGreen.at<float>(i);
        if(sum >= totalPixel * HighCut * 0.01){
            MaxGreen = i;
            break;
        }
    }
    int minn = min(MinBlue, min(MinGreen, MinRed));
    int maxx = max(MaxBlue, max(MaxGreen, MaxRed));
    //自动对比度
    for(int i = 0; i < 256; i++){
        if(i <= minn){
            Pixel[i*3+2] = 0;
        }else{
            if(i > maxx){
                Pixel[i*3+2] = 255;
            }else{
                float temp = (float)(i - minn) / (maxx - minn);
                Pixel[i*3+2] = (uchar)(temp*255);
            }
        }
        if(i <= minn){
            Pixel[i*3+1] = 0;
        }else{
            if(i > maxx){
                Pixel[i*3+1] = 255;
            }else{
                float temp = (float)(i - minn) / (maxx - minn);
                Pixel[i*3+1] = (uchar)(temp*255);
            }
        }
        if(i <= minn){
            Pixel[i*3] = 0;
        }else{
            if(i > maxx){
                Pixel[i*3] = 255;
            }else{
                float temp = (float)(i - minn) / (maxx - minn);
                Pixel[i*3] = (uchar)(temp*255);
            }
        }
    }
    Mat dst;
    Mat TMP(1, 256, CV_8UC3, Pixel);
    LUT(src, TMP, dst);
    return dst;
}

int main(){
    Mat src = imread("../3.png");
    Mat dst = AutoLevel(src, 0.005, 0.005);
    imshow("origin", src);
    imshow("result", dst);
    waitKey(0);
}