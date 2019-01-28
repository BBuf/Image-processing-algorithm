#include "opencv2/opencv.hpp"
#include "iostream"
#include "algorithm"
#include "vector"
using namespace std;
using namespace cv;

const int g_ndMaxValue = 100;
const int g_nsigmaColorMaxValue = 200;
const int g_nsigmaSpaceMaxValue = 200;
int g_ndValue, g_nsigmaColorValue, g_nsigmaSpaceValue;
Mat src, dst;
void on_bilateralFilterTrackbar(int, void*){
    bilateralFilter(src, dst, g_ndValue, g_nsigmaColorValue, g_nsigmaSpaceValue);
    imshow("filtering", dst);
}

int main(){
//    Mat src = imread("/home/streamax/CLionProjects/Paper/101507_588686279_15.jpg");
//    Mat dst;
//    bilateralFilter(src, dst, 9, 50, 50);
//    imshow("src", src);
//    imshow("dst", dst);
//    imwrite("../result.jpg", dst);
//    waitKey(0);
    src = imread("../101507_588686279_15.jpg");
    namedWindow("origin", WINDOW_AUTOSIZE);
    imshow("origin", src);
    g_ndValue = 9;
    g_nsigmaColorValue = 50;
    g_nsigmaSpaceValue = 50;
    namedWindow("filtering", WINDOW_AUTOSIZE);
    char dname[20];
    sprintf(dname, "Neighborhood diamter %d", g_ndMaxValue);
    char sigmaColorName[20];
    sprintf(sigmaColorName, "sigmaColor %d", g_nsigmaColorMaxValue);
    char sigmaSpaceName[20];
    sprintf(sigmaSpaceName, "sigmaSpace %d", g_nsigmaSpaceMaxValue);
    //创建轨迹条
    createTrackbar(dname, "filtering", &g_ndValue, g_ndMaxValue, on_bilateralFilterTrackbar);
    on_bilateralFilterTrackbar(g_ndValue, 0);
    createTrackbar(sigmaColorName, "filtering", &g_nsigmaColorValue, g_nsigmaColorMaxValue, on_bilateralFilterTrackbar);
    on_bilateralFilterTrackbar(g_nsigmaColorValue, 0);
    createTrackbar(sigmaSpaceName, "filtering", &g_nsigmaSpaceValue, g_nsigmaSpaceMaxValue, on_bilateralFilterTrackbar);
    on_bilateralFilterTrackbar(g_nsigmaSpaceValue, 0);
    waitKey(0);
    return 0;
}