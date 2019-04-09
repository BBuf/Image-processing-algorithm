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
    Mat gainArr = meanGlobal / varLocal; //增益系数矩阵
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
    int C = 150;
    int n = 5;
    float MaxCG = 3;
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