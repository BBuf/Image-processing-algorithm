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

#define PI 3.14159265
Mat work(Mat src){
    int row = src.rows;
    int col = src.cols;
    Mat dst(row, col, CV_8UC3);
    //RGB2HSI
    Mat H = Mat(row, col, CV_64FC1);
    Mat S = Mat(row, col, CV_64FC1);
    Mat I = Mat(row, col, CV_64FC1);
    int mp[256]={0};
    double mp2[256]={0.0};
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            double h, s, newi, th;
            double B = (double)src.at<Vec3b>(i, j)[0] / 255.0;
            double G = (double)src.at<Vec3b>(i, j)[1] / 255.0;
            double R = (double)src.at<Vec3b>(i, j)[2] / 255.0;
            double mi, mx;
            if(R > G && R > B){
                mx = R;
                mi = min(G, B);
            }
            else{
                if(G > B){
                    mx = G;
                    mi = min(R, B);
                }else{
                    mx = B;
                    mi = min(R, G);
                }
            }
            newi = (R + G + B) / 3.0;
            if(newi < 0) newi = 0;
            else if(newi > 1) newi = 1.0;
            if(newi == 0 || mx == mi){
                s = 0;
                h = 0;
            }else{
                s = 1 - mi / newi;
                th = (R - G) * (R - G) + (R - B) * (G - B);
                th = sqrt(th)+1e-5;
                th = acos(((R-G+R-B)*0.5)/th);
                if(G >= B) h = th;
                else h = 2 * PI - th;
            }
            h = h / (2*PI);
            H.at<double>(i, j) = h;
            S.at<double>(i, j) = s;
            I.at<double>(i, j) = newi;
            mp[(int)((src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2]) / 3)]++;
        }
    }
    for(int i = 0; i < 256; i++){
        mp2[i] = (double)mp[i] / (double)(row * col);
    }
    double mI = 0;
    for(int i = 0; i < 256; i++){
        mI += (i / 255.0)  * mp2[i];
    }
    printf("mI: %.5f\n", mI);
    double var = 0;
    double ThresHold = 0;
    for(int i = 0; i < 256; i++){
        double T = 1.0 * i / 256;
        double P1 = 0.0;
        double mT = 0.0;
        for(int j = 0; j <= i; j++){
            P1 += mp2[j];
            mT += (double)(j / 255.0) * mp2[j];
        }
        if(P1 == 0) continue;
        if(((mI*P1 - mT)*(mI*P1 - mT) / (P1*(1-P1))) > var){
            var = (mI*P1 - mT)*(mI*P1 - mT) / (P1*(1-P1));
            ThresHold = T;
            //printf("%d %.5f\n", i, ThresHold);
        }
    }
    printf("Thres: %.5f\n", ThresHold);
    int k = 50;
    int cnt = 0;
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            if(I.at<double>(i, j) <= ThresHold){
                cnt++;
            }
        }
    }
    printf("cnt: %d\n", cnt);
    double A = (double)k * sqrt((double)cnt / (double)(row * col - cnt));
    printf("A: %.5f\n", A);
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            double D, C;
            if(I.at<uchar>(i, j) <= ThresHold){
                D = A;
                C = 1.0 / log2(D+1);
            }else{
                D = (double)(ThresHold * A - ThresHold) / double((1 - ThresHold) * (I.at<double>(i, j))) - (double)(ThresHold * A - 1) / (1 - ThresHold);
                C = 1.0 / log2(D+1);
            }
            I.at<double>(i, j) = (C * log2(D * (double)I.at<double>(i, j) + 1));
        }
    }
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            double preh = H.at<double>(i, j);
            double pres = S.at<double>(i, j);
            double prei = I.at<double>(i, j);
           // printf("H: %.5f S: %.5f I: %.5f\n", preh, pres, prei);
            double r = 0, g = 0, b = 0;
            if(prei == 0.0){
                r = 0;
                g = 0;
                b = 0;
            }
            else{
                if(pres == 0.0){
                    r = g = b = prei;
                }
                double t1, t2, t3;
                t1 = (1.0 - pres) / 3.0;
                t2 = pres * cos(preh);
                if(preh >= 0 && preh < (PI * 2 / 3)){
                    b = t1;
                    t3 = cos(PI/3 - preh);
                    r = (1 + t2 / t3) / 3;
                    g = 1.0 - r - b;
                    r = 3 * prei * r;
                    g = 3 * g * prei;
                    b = 3 * prei * b;
                }else if(preh >= (PI * 2 / 3) && preh < (PI * 4 / 3)){
                    r = t1;
                    t3 = cos(PI - preh);
                    g = (1 + t2 / t3) / 3;
                    b = 1 - r - g;
                    r = 3 * prei * r;
                    g = 3 * g * prei;
                    b = 3 * prei * b;
                }else if(preh >= (PI * 4 / 3) && preh < (PI * 2)){
                    g = t1;
                    t3 = cos(PI * 5 / 3 - preh);
                    b = (1 + t2 / t3) / 3;
                    r = 1 - g - b;
                    r = 3 * prei * r;
                    g = 3 * g * prei;
                    b = 3 * prei * b;
                }
            }
            //printf("%d %d %d\n", (int)(r*255), (int)(g*255), (int)(b*255));
            dst.at<Vec3b>(i, j)[0] = (int)(b*255);
            dst.at<Vec3b>(i, j)[1] = (int)(g*255);
            dst.at<Vec3b>(i, j)[2] = (int)(r*255);
        }
    }
    return dst;
}

//自适应对比度增强算法，C表示对高频的直接增益系数,n表示滤波半径，maxCG表示对CG做最大值限制
Mat ACE(Mat src, int C = 3, int n = 3, float MaxCG = 7.5){
    int row = src.rows;
    int col = src.cols;
    Mat meanLocal; //图像局部均值
    Mat varLocal; //图像局部方差
    Mat meanGlobal; //全局均值
    Mat varGlobal; //全局标准差
    blur(src.clone(), meanLocal, Size(n, n));
    Mat highFreq = src - meanLocal;
    varLocal = highFreq.mul(highFreq);
    varLocal.convertTo(varLocal, CV_32F);
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            varLocal.at<float>(i, j) = (float)sqrt(varLocal.at<float>(i, j));
        }
    }
    meanStdDev(src, meanGlobal, varGlobal);
    Mat gainArr = varGlobal / varLocal; //增益系数矩阵
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            if(gainArr.at<float>(i, j) > MaxCG){
                gainArr.at<float>(i, j) = MaxCG;
            }
        }
    }
    printf("%d %d\n", row, col);
    gainArr.convertTo(gainArr, CV_8U);
    gainArr = gainArr.mul(highFreq);
    Mat dst1 = meanLocal + gainArr;
    Mat dst2 = meanLocal + C * highFreq;
    return dst1;
}

int main(){
    Mat src = imread("../test.png");
    vector <Mat> now;
    split(src, now);
    int C = 4;
    int n = 50;
    float MaxCG = 5;
    Mat dst1 = ACE(now[0], C, n, MaxCG);
    Mat dst2 = ACE(now[1], C, n, MaxCG);
    Mat dst3 = ACE(now[2], C, n, MaxCG);
    now.clear();
    Mat dst;
    now.push_back(dst1);
    now.push_back(dst2);
    now.push_back(dst3);
    cv::merge(now, dst);
    imshow("origin", src);
    imshow("result", dst);
    imwrite("../result.jpg", dst);
    waitKey(0);
    return 0;
}