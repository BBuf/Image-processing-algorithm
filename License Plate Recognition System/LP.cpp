#include <LP.h>
#include <stdio.h>
#include <iostream>
#include <io.h>
#include "MyFrame.h"
#include <wx/wx.h>
#include <wx/dcmemory.h>
#include <unordered_map>
#include <wx/wx.h>
#include <wx/dcclient.h>
#include <wx/dcmemory.h>
#include <vector>
#include <algorithm>
#include <wx/textdlg.h>
#include <wx/dcbuffer.h>
#include "sstream"

using namespace cv;
using namespace std;
bool global_flag;

void RgbConvToGray(const Mat &inputImage, Mat &outpuImage) //g = 0.3R+0.59G+0.11B
{
    outpuImage = Mat(inputImage.rows, inputImage.cols, CV_8UC1);
    for (int i = 0; i < inputImage.rows; ++i) {
        uchar *ptrGray = outpuImage.ptr<uchar>(i);
        const Vec3b *ptrRgb = inputImage.ptr<Vec3b>(i);
        for (int j = 0; j < inputImage.cols; ++j) {
            ptrGray[j] = 0.3 * ptrRgb[j][2] + 0.59 * ptrRgb[j][1] + 0.11 * ptrRgb[j][0];
        }
    }
}

void posDetect_closeImage(Mat &inputImage, vector<RotatedRect> &rects) { //初步找到候选区域rects
    Mat canny_img, threshold_img;
    Canny(inputImage, canny_img, 150, 220);//第三个参数和第四个参数表示阈值，这二个阈值中当中的小阈值用来控制边缘连接，
    // 大的阈值用来控制强边缘的初始分割即如果一个像素的梯度大与上限值，则被认为是边缘像素，如果小于下限阈值，则被抛弃。如果该
    // 点的梯度在两者之间则当这个点与高于上限值的像素点连接时我们才保留，否则删除。
    threshold(canny_img, threshold_img, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);//otsu自动获得阈值
    Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));//返回指定形状和大小的结构元素
    morphologyEx(threshold_img, threshold_img, CV_MOP_CLOSE, element);//形态学闭操作
    morphologyEx(threshold_img, threshold_img, MORPH_OPEN, element);//形态学开操作
    //寻找区域外轮廓
    vector<vector<Point> > ans;
    findContours(threshold_img, ans, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); //只检测外轮廓
    //对候选的轮廓进行进一步筛选
    vector<vector<Point>>::iterator it = ans.begin();
    while (it != ans.end()) {
        RotatedRect temp = minAreaRect(Mat(*it)); //返回每个轮廓的最小有界矩形区域
        if (!needsize_closeImage(temp)) {
            it = ans.erase(it);
        } else {
            rects.push_back(temp);
            it++;
        }
    }
}

bool needsize_closeImage(const RotatedRect &candidate) {
    float error = 0.4;
    const float aspect = 44 / 14; //长宽比
    int min = 20 * aspect * 20;
    int max = 180 * aspect * 180;
    float rmin = aspect - aspect * error;
    float rmax = aspect + aspect * error;
    int area = candidate.size.height * candidate.size.width;
    float r = (float) candidate.size.width / (float) candidate.size.height;
    if (r < 1) r = 1 / r;
    if ((area < min || area > max) || (r < rmin || r > rmax)) return false;
    else return true;
}

void posDetect(Mat &inputImage, vector<RotatedRect> &rects) {//初步找到候选区域 rects
    Mat sobel_img, threshold_img;
    Sobel(inputImage, sobel_img, CV_8U, 1, 0, 3, 1, 0);
    threshold(sobel_img, threshold_img, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
    Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));//返回指定形状和大小的结构元素
    morphologyEx(threshold_img, threshold_img, CV_MOP_CLOSE, element);//形态学闭操作
    morphologyEx(threshold_img, threshold_img, MORPH_OPEN, element);//形态学开操作
    //寻找区域外轮廓
    vector<vector<Point> > ans;
    findContours(threshold_img, ans, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); //只检测外轮廓
    //对候选的轮廓进行进一步筛选
    vector<vector<Point>>::iterator it = ans.begin();
    while (it != ans.end()) {
        RotatedRect temp = minAreaRect(Mat(*it)); //返回每个轮廓的最小有界矩形区域
        if (!needsize(temp)) {
            it = ans.erase(it);
        } else {
            rects.push_back(temp);
            it++;
        }
    }
}

void svm_train(CvSVM &svmClassfier) {
    FileStorage fs;
    fs.open("data/SVM.xml", FileStorage::READ);
    Mat SVM_TrainningData;
    Mat SVM_Classes;
    fs["TrainingData"] >> SVM_TrainningData;
    fs["classes"] >> SVM_Classes;
    CvSVMParams SVM_params;
    SVM_params.kernel_type = CvSVM::LINEAR;
    svmClassfier.train(SVM_TrainningData, SVM_Classes, Mat(), Mat(), SVM_params);
    fs.release();
    printf("train success!\n");
}


bool needsize(const RotatedRect &candidate) {
    float error = 0.4;
    const float aspect = 44 / 14; //长宽比
    int min = 20 * aspect * 20;
    int max = 180 * aspect * 180;
    float rmin = aspect - 2 * aspect * error;
    float rmax = aspect + 2 * aspect * error;
    int area = candidate.size.height * candidate.size.width;
    float r = (float) candidate.size.width / (float) candidate.size.height;
    if (r < 1) r = 1 / r;
    if ((area < min || area > max) || (r < rmin || r > rmax)) return false;
    else return true;
}

void optimPosDetect(vector<RotatedRect> &rects_sImg, vector<RotatedRect> &rects_grayImage,
                    vector<RotatedRect> &rects_closeImg,
                    vector<RotatedRect> &rects_optimal) {
    for (int i = 0; i < rects_sImg.size(); i++) {
        for (int j = 0; j < rects_grayImage.size(); j++) {
            if (calOverlap(rects_sImg[i].boundingRect(), rects_grayImage[j].boundingRect()) > 0.2) {
                if (rects_sImg[i].boundingRect().width * rects_sImg[i].boundingRect().height >=
                    rects_grayImage[j].boundingRect().width * rects_grayImage[j].boundingRect().height) {
                    rects_optimal.push_back(rects_sImg[i]);
                } else {
                    rects_optimal.push_back(rects_grayImage[j]);
                }
            }
        }
    }
    for (int i = 0; i < rects_optimal.size(); i++) {
        for (int j = 0; j < rects_closeImg.size(); j++) {
            if ((calOverlap(rects_optimal[i].boundingRect(), rects_closeImg[j].boundingRect()) < 0.2) &&
                (calOverlap(rects_optimal[i].boundingRect(), rects_closeImg[j].boundingRect()) > 0.05)) {
                rects_optimal.push_back(rects_closeImg[j]);
            }
        }
    }
}

bool check_closeImage(const RotatedRect &candidate) {
    float error = 0.4;
    const float standrad_ratio = 44 / 14;
    int min = 20 * standrad_ratio * 20;
    int max = 180 * standrad_ratio * 180;
    float rmin = standrad_ratio - standrad_ratio * error;
    float rmax = standrad_ratio + standrad_ratio * error;
    int area = candidate.size.height * candidate.size.width;
    float r = (float) candidate.size.width / (float) candidate.size.height;
    if (r < 1) r = 1 / r;
    if ((area < min || area > max) || (r < rmin || r > rmax)) return false;
    else return true;
}


float calOverlap(const Rect &box1, const Rect &box2) {
    if (box1.x > box2.x + box2.width) return 0.0;
    if (box1.y > box2.y + box2.height) return 0.0;
    if (box1.x + box1.width < box2.x) return 0.0;
    if (box1.y + box1.height < box2.y) return 0.0;//这4个语句是要限制两个矩阵必须在一起才有叠加性
    float colInt = min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x);
    float rowInt = min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y);
    float intersection = colInt * rowInt;//重叠矩形的大小
    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    return intersection / (area1 + area2 - intersection);//返回的值是重叠面积占俩个面积的交集比例
}

void GetStandardPlate(Mat &inputImg, vector<RotatedRect> &rects_optimal, vector<Mat> &output_area) {
    float r, angle;
    for (int i = 0; i < rects_optimal.size(); i++) {
        //旋转区域
        angle = rects_optimal[i].angle;
        r = (float) rects_optimal[i].size.width / (float) rects_optimal[i].size.height;
        if (r < 1) angle = 90 + angle; //旋转图像使得其得到的长大于高度图像
        Mat rotmat = getRotationMatrix2D(rects_optimal[i].center, angle, 1); //获得变矩形对象
        Mat rotated_img;
        warpAffine(inputImg, rotated_img, rotmat, inputImg.size(), CV_INTER_CUBIC);
        Size rect_size = rects_optimal[i].size;
        if (r < 1) {
            swap(rect_size.width, rect_size.height);
        }
        Mat crop_img;
        getRectSubPix(rotated_img, rect_size, rects_optimal[i].center, crop_img);
        //用光照直方图调整所有裁剪得到的图像，使其具有相同的宽度和高度，适用于训练和分类
        Mat result;
        result.create(33, 144, CV_8UC3);
        resize(crop_img, result, result.size(), 0, 0, INTER_CUBIC);
        Mat gray_result;
        RgbConvToGray(result, gray_result);
        equalizeHist(gray_result, gray_result);//直方图均衡化
        output_area.push_back(gray_result);
    }
}


void char_segment(const Mat &inputImg, vector<Mat> &dst_mat) {//得到20*20的标准字符分割图像
    Mat threshold_img;
    threshold(inputImg, threshold_img, 180, 255, CV_THRESH_BINARY);//180是阈值,255是高于阈值设成的值,最终得到黑白图像
    Mat result_img;
    threshold_img.copyTo(result_img);
    clearLiuDing(result_img);
    Mat result2;
    inputImg.copyTo(result2);
    vector<vector<Point> > contours;
    findContours(result_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    vector<vector<Point> >::iterator it = contours.begin();
    vector<RotatedRect> char_rects;
    drawContours(result2, contours, -1, Scalar(0, 255, 255), 1);
    while (it != contours.end()) {
        RotatedRect minArea = minAreaRect(Mat(*it));
        Point2f vertices[4];
        minArea.points(vertices);
        if (!char_check(minArea)) {
            it = contours.erase(it);
        } else {
            ++it;
            char_rects.push_back(minArea);
        }
    }
    char_sort(char_rects);
    vector<Mat> char_mat;
    int char_number = char_rects.size();
    vector<float> x_distance;
    for (int i = 0; i < char_number; i++) {
        x_distance.push_back(char_rects[i].center.x);
    }
    vector<Point2f> rect_points;
    for (int z = 0; z < char_rects.size(); z++) {
        Point2f vertex[4];
        char_rects[z].points(vertex);
        for (int i = 0; i < 4; i++) {
            rect_points.push_back(vertex[i]);
        }
    }
    float x_min[20];
    float x_max[20];
    float y_min[20];
    float y_max[20];
    for (int z = 0; z < char_rects.size(); z++) {
        float min_x = rect_points[z * 4].x;
        float min_y = rect_points[z * 4].y;
        float max_x = 0;
        float max_y = 0;
        for (int w = 0; w < 4; w++) {
            min_x = min_x < rect_points[z * 4 + w].x ? min_x : rect_points[z * 4 + w].x;
            max_x = max_x > rect_points[z * 4 + w].x ? max_x : rect_points[z * 4 + w].x;
            min_y = min_y < rect_points[z * 4 + w].y ? min_y : rect_points[z * 4 + w].y;
            max_y = max_y > rect_points[z * 4 + w].y ? max_y : rect_points[z * 4 + w].y;
        }
        x_min[z] = min_x;
        x_max[z] = max_x;
        y_min[z] = min_y;
        y_max[z] = max_y;
    }
    vector<Point2f> real_points;
    for (int z = 0; z < char_rects.size(); z++) {
        Point2f a, b, c, d;
        a.x = x_min[z];
        a.y = y_max[z];
        b.x = x_min[z];
        b.y = y_min[z];
        c.x = x_max[z];
        c.y = y_min[z];
        d.x = x_max[z];
        d.y = y_max[z];
        real_points.push_back(a);
        real_points.push_back(b);
        real_points.push_back(c);
        real_points.push_back(d);
    }
    vector<Point2f> conbine;
    int flag1 = 0; //这是合并的标志，每一次进行合并的时候，进行加1；
    for (int center_number = 0; center_number < 5; center_number++) {
        int distanst = x_distance[center_number + 1] - x_distance[center_number];
        int distanst2 = x_distance[center_number + 2] - x_distance[center_number + 1];
        if (distanst <= 8 && distanst2 >= 8) {
            conbine.push_back(real_points[center_number * 4]);
            conbine.push_back(real_points[center_number * 4 + 1]);
            conbine.push_back(real_points[center_number * 4 + 6]);
            conbine.push_back(real_points[center_number * 4 + 7]);
            center_number += 1;
            flag1 += 1;
        }
        if (distanst2 <= 8 && distanst2 <= 8) {
            conbine.push_back(real_points[center_number * 4]);
            conbine.push_back(real_points[center_number * 4 + 1]);
            conbine.push_back(real_points[center_number * 4 + 10]);
            conbine.push_back(real_points[center_number * 4 + 11]);
            center_number += 2;
            flag1 += 2;
        } else {
            conbine.push_back(real_points[center_number * 4]);
            conbine.push_back(real_points[center_number * 4 + 1]);

            conbine.push_back(real_points[center_number * 4 + 2]);
            conbine.push_back(real_points[center_number * 4 + 3]);
        }
    }
    int s = (conbine.size()) / 4 + flag1;
    int s1 = (real_points.size()) / 4;
    for (; s < s1; s++) {
        conbine.push_back(real_points[s * 4]);
        conbine.push_back(real_points[s * 4 + 1]);
        conbine.push_back(real_points[s * 4 + 2]);
        conbine.push_back(real_points[s * 4 + 3]);
    }
    Mat char_img;
    int char_rects_conbin_number = char_rects.size() - flag1;
    for (int i = 0; i < char_rects_conbin_number; i++) {
        char_img = inputImg(Range(conbine[i * 4 + 1].y, conbine[i * 4].y),
                            Range(conbine[i * 4 + 1].x, conbine[i * 4 + 2].x));//首先这个
        char_mat.push_back(char_img);
    }
    for (int i = 0; i < char_rects.size(); i++) {
        char_mat.push_back(Mat(threshold_img, char_rects[i].boundingRect()));
    }
    if (char_mat.size() < 7) {
        global_flag = 0;
        return;
    }
    dst_mat.resize(7);
    for (int i = 0; i < 7; i++) {
        if (i == 0) {
            dst_mat[i] = char_mat[i];
            resize(dst_mat[i], dst_mat[i], Size(40, 28));
        } else {
            dst_mat[i] = char_mat[i];
            resize(dst_mat[i], dst_mat[i], Size(28, 28));
        }
    }
//    Mat train_mat(2,3,CV_32FC1);
//    int length;
//    dst_mat.resize(7);//车牌只有7个字符
//    Point2f srcTri[3], dstTri[3];
//    srcTri[0] = Point2f(0,0);
//    srcTri[1] = Point2f(char_mat[0].cols-1,0);
//    srcTri[2] = Point2f(0, char_mat[0].rows-1);
//    length = char_mat[0].rows>char_mat[0].cols?char_mat[0].rows:char_mat[0].cols;
//    dstTri[0] = Point2f(0.0,0.0);
//    dstTri[1] = Point2f(length,0);
//    dstTri[2] = Point2f(0, length);
//    train_mat = getAffineTransform(srcTri, dstTri);//仿射矩阵
//    dst_mat[0] = Mat::zeros(length,length,char_mat[0].type());
//    warpAffine(char_mat[0],dst_mat[0],train_mat,dst_mat[0].size(),INTER_LINEAR,BORDER_CONSTANT,Scalar(0));//仿射变换
//    resize(dst_mat[0],dst_mat[0],Size(32,40),INTER_CUBIC);
//    Mat element = getStructuringElement(MORPH_RECT, Size(1, 1));
//    erode(dst_mat[0], dst_mat[0], element);
//    for(int i=1; i<7; i++){
//        srcTri[0] = Point2f(0,0);
//        srcTri[1] = Point2f(char_mat[i].cols-1,0);
//        srcTri[2] = Point2f(0, char_mat[i].rows-1);
//        length = char_mat[i].rows>char_mat[i].cols?char_mat[i].rows:char_mat[i].cols;
//        dstTri[0] = Point2f(0,0);
//        dstTri[1] = Point2f(length,0);
//        dstTri[2] = Point2f(0, length);
//        train_mat = getAffineTransform(srcTri, dstTri);//仿射矩阵
//        dst_mat[i] = Mat::zeros(length,length,char_mat[i].type());
//        warpAffine(char_mat[i],dst_mat[i],train_mat,dst_mat[i].size(),INTER_LINEAR,BORDER_CONSTANT,Scalar(0));//仿射变换
//        resize(dst_mat[i],dst_mat[i],Size(28,28),INTER_CUBIC);
//        Mat element = getStructuringElement(MORPH_RECT, Size(1, 1));
//        erode(dst_mat[i], dst_mat[i], element);
//    }
}

bool char_check(const RotatedRect &src) {
    float width, height;
    if (src.size.width >= src.size.height) {
        width = (float) src.size.height, height = (float) src.size.width;
    } else {
        height = (float) src.size.height, width = (float) src.size.width;
    }

    float cur_ratio = width / height, minGao = 15, maxGao = 35, minRatio = 0.00, maxRatio = 1.0;
    if (cur_ratio > minRatio && cur_ratio < maxRatio && height > minGao && height < maxGao) return true;
    else return false;
}


void char_sort(vector<RotatedRect> &in_char) {//对字符区域排序
    vector<RotatedRect> out_char;
    const int length = in_char.size();
    int index[length];
    for (int i = 0; i < length; i++) {
        index[i] = i;
    }
    float centerX[length];
    for (int i = 0; i < length; i++) {
        centerX[i] = in_char[i].center.x;
    }
    for (int j = 0; j < length; j++) {//冒泡排序
        for (int i = length - 2; i >= j; i--) {
            if (centerX[i] > centerX[i + 1]) {
                swap(centerX[i], centerX[i + 1]);
                swap(index[i], index[i + 1]);
            }
        }
    }
    for (int i = 0; i < length; i++) {
        out_char.push_back(in_char[(index[i])]);
    }
    in_char.clear();//清空in_char
    in_char = out_char;//把排序好的字符区域的向量重新复制给in_char
}


void clearLiuDing(Mat &src) {
    for (int i = 0; i < src.rows; i++) {
        if (i < 2 || i > (src.rows - 2)) {
            for (int j = 0; j < src.cols; j++) src.at<char>(i, j) = 0;
        }
    }
    for (int i = 0; i < src.cols; i++) {
        if (i < 2 || i > (src.cols - 2)) {
            for (int j = 0; j < src.rows; j++) src.at<char>(j, i) = 0;
        }
    }
}

void hsvImageBlue(Mat &hsvImage) {
    int channels = hsvImage.channels();
    int imageRows = hsvImage.rows;
    int imageCols = hsvImage.cols * channels;
    if (hsvImage.isContinuous()) {
        imageCols *= imageRows;
        imageRows = 1;
    }
    int lowcolor = 100;
    int highcolor = 140;
    uchar *now;
    for (int i = 0; i < imageRows; i++) {
        now = hsvImage.ptr<uchar>(i);
        for (int j = 0; j < imageCols; j += 3) {
            int nowH = (int) now[j], nowS = (int) now[j + 1], nowV = now[j + 2];
            bool success = false;
            if (nowH > lowcolor && nowH < highcolor && nowS > 70 && nowS < 255 && nowV > 70 && nowV < 255) {
                success = true;
            }
            if (success) {
                now[j] = 0, now[j + 1] = 0, now[j + 2] = 255;
            } else {
                now[j] = 0, now[j + 1] = 0, now[j + 2] = 0;
            }
        }
    }
}

void hsvImageYello(Mat &hsvImage) {
    int channels = hsvImage.channels();
    int imageRows = hsvImage.rows;
    int imageCols = hsvImage.cols * channels;
    if (hsvImage.isContinuous()) {
        imageCols *= imageRows;
        imageRows = 1;
    }
    int lowcolor = 15;
    int highcolor = 40;
    uchar *now;
    for (int i = 0; i < imageRows; i++) {
        now = hsvImage.ptr<uchar>(i);
        for (int j = 0; j < imageCols; j += 3) {
            int nowH = (int) now[j], nowS = (int) now[j + 1], nowV = now[j + 2];
            bool success = false;
            if (nowH > lowcolor && nowH < highcolor && nowS > 70 && nowS < 255 && nowV > 70 && nowV < 255) {
                success = true;
            }
            if (success) {
                now[j] = 0, now[j + 1] = 0, now[j + 2] = 255;
            } else {
                now[j] = 0, now[j + 1] = 0, now[j + 2] = 0;
            }
        }
    }
}

bool solve1() {
    char filename[200];
    sprintf(filename, "saveImage/1.jpg");
    Mat input_img = imread(filename, 1);
    Mat input_img2;
    input_img2 = input_img.clone();
    if (input_img.empty()) {
        return false;
    }
    Mat hsvImg;
    vector<Mat> hsvSplit;
    cvtColor(input_img, hsvImg, CV_BGR2HSV);//把图像转为HSV模型
    split(hsvImg, hsvSplit);
    equalizeHist(hsvSplit[2], hsvSplit[2]);
    merge(hsvSplit, hsvImg);//对彩色直方图做均衡化，因为读取的是彩色图,直方图均衡化需要在HSV空间进行
    hsvImageBlue(hsvImg);
    Mat grey_src, threshold_img;
    vector<Mat> hsvSplit_done;
    split(hsvImg, hsvSplit_done);
    grey_src = hsvSplit_done[2];
    vector<RotatedRect> rects;
    Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));//闭形态学的结构元素
    morphologyEx(grey_src, threshold_img, CV_MOP_CLOSE, element);
    morphologyEx(threshold_img, threshold_img, MORPH_OPEN, element); //形态学处理
    //寻找车牌区域的轮廓
    vector<vector<Point> > contours;
    findContours(threshold_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); //只检测外面的轮廓
    //对候选的轮廓进行进一步的筛选
    vector<vector<Point> >::iterator it = contours.begin();
    while (it != contours.end()) {
        RotatedRect temp = minAreaRect(Mat(*it)); //返回每个轮廓的最小有界矩形区域
        if (!check_closeImage(temp)) {
            it = contours.erase(it);
        } else {
            rects.push_back(temp);
            ++it;
        }
    }
    Mat result;
    input_img.copyTo(result);
    drawContours(result, contours, -1, Scalar(0, 0, 255), 1);
    vector<Mat> output_area;
    GetStandardPlate(input_img, rects, output_area); //获得144×33的候选车牌区域
    CvSVM svmClassifyModel;
    svm_train(svmClassifyModel);
    vector<Mat> plates_svm;
    for (int i = 0; i < output_area.size(); i++) {
        Mat img = output_area[i];
        Mat now = img.reshape(1, 1);
        now.convertTo(now, CV_32FC1);
        int res = (int) svmClassifyModel.predict(now);
        if (res == 1)
            plates_svm.push_back(output_area[i]);
    }
    if (plates_svm.size() == 0) {
        return solve3();
    }
    //cout<<plates_svm.size()<<endl;
    vector<Mat> char_seg;
    char_segment(plates_svm[0], char_seg);
//    cv::imshow("char0", char_seg[0]);
//    waitKey(0);
//    cv::imshow("char1", char_seg[1]);
//    waitKey(0);
//    cv::imshow("char2", char_seg[2]);
//    waitKey(0);
//    cv::imshow("char3", char_seg[3]);
//    waitKey(0);
//    cv::imshow("char4", char_seg[4]);
//    waitKey(0);
//    cv::imshow("char5", char_seg[5]);
//    waitKey(0);
//    cv::imshow("char6", char_seg[6]);
//    waitKey(0);
    cv::imwrite("python/zxytest_images/1.jpg", char_seg[0]);
    cv::imwrite("python/zxytest_images/2.jpg", char_seg[1]);
    cv::imwrite("python/zxytest_images/3.jpg", char_seg[2]);
    cv::imwrite("python/zxytest_images/4.jpg", char_seg[3]);
    cv::imwrite("python/zxytest_images/5.jpg", char_seg[4]);
    cv::imwrite("python/zxytest_images/6.jpg", char_seg[5]);
    cv::imwrite("python/zxytest_images/7.jpg", char_seg[6]);
    destroyAllWindows();
    return true;
}

void getfiles(string path, vector<string> &files) {
    //文件句柄
    long hFile = 0;
    //文件信息
    struct _finddata_t fileinfo;
    string p;
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo))
        != -1) {
        do {
            //如果是目录,迭代之
            //如果不是,加入列表
            if ((fileinfo.attrib & _A_SUBDIR)) {
                if (strcmp(fileinfo.name, ".") != 0
                    && strcmp(fileinfo.name, "..") != 0)
                    getfiles(p.assign(path).append("\\").append(fileinfo.name),
                             files);
            } else {
                files.push_back(
                        p.assign(path).append("\\").append(fileinfo.name));
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}

int getFileNum(string x) {
    fstream file;
    vector<string> files;
    string filepath = x;
    getfiles(filepath, files);
    int size = files.size();
    return size;
}

bool solve2() {
    char filename[200];
    sprintf(filename, "saveImage/1.jpg");
    Mat input_img = imread(filename, 1);
    Mat input_img2;
    input_img2 = input_img.clone();
    if (input_img.empty()) {
    }
    Mat hsvImg;
    vector<Mat> hsvSplit;
    cvtColor(input_img, hsvImg, CV_BGR2HSV);//把图像转为HSV模型
    split(hsvImg, hsvSplit);
    equalizeHist(hsvSplit[2], hsvSplit[2]);
    merge(hsvSplit, hsvImg);//对彩色直方图做均衡化，因为读取的是彩色图,直方图均衡化需要在HSV空间进行
    hsvImageBlue(hsvImg);
    Mat grey_src, threshold_img;
    vector<Mat> hsvSplit_done;
    split(hsvImg, hsvSplit_done);
    grey_src = hsvSplit_done[2];
    vector<RotatedRect> rects;
    Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));//闭形态学的结构元素
    morphologyEx(grey_src, threshold_img, CV_MOP_CLOSE, element);
    morphologyEx(threshold_img, threshold_img, MORPH_OPEN, element); //形态学处理
    //寻找车牌区域的轮廓
    vector<vector<Point> > contours;
    findContours(threshold_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); //只检测外面的轮廓
    //对候选的轮廓进行进一步的筛选
    vector<vector<Point> >::iterator it = contours.begin();
    while (it != contours.end()) {
        RotatedRect temp = minAreaRect(Mat(*it)); //返回每个轮廓的最小有界矩形区域
        if (!check_closeImage(temp)) {
            it = contours.erase(it);
        } else {
            rects.push_back(temp);
            ++it;
        }
    }
    Mat result;
    input_img.copyTo(result);
    drawContours(result, contours, -1, Scalar(0, 0, 255), 1);
    vector<Mat> output_area;
    GetStandardPlate(input_img, rects, output_area); //获得144×33的候选车牌区域
    CvSVM svmClassifyModel;
    svm_train(svmClassifyModel);
    vector<Mat> plates_svm;
    for (int i = 0; i < output_area.size(); i++) {
        Mat img = output_area[i];
        Mat now = img.reshape(1, 1);
        now.convertTo(now, CV_32FC1);
        int res = (int) svmClassifyModel.predict(now);
        if (res == 1)
            plates_svm.push_back(output_area[i]);
    }
    if (plates_svm.size() == 0) {
        return solve3();
    }
    //cout<<plates_svm.size()<<endl;
    cv::imshow("Car Locate", plates_svm[0]);
    waitKey(0);
    vector<Mat> char_seg;
    char_segment(plates_svm[0], char_seg);
    cv::imshow("char0", char_seg[0]);
    waitKey(0);
    cv::imshow("char1", char_seg[1]);
    waitKey(0);
    cv::imshow("char2", char_seg[2]);
    waitKey(0);
    cv::imshow("char3", char_seg[3]);
    waitKey(0);
    cv::imshow("char4", char_seg[4]);
    waitKey(0);
    cv::imshow("char5", char_seg[5]);
    waitKey(0);
    cv::imshow("char6", char_seg[6]);
    waitKey(0);
    cv::imwrite("python/zxytest_images/1.jpg", char_seg[0]);
    cv::imwrite("python/zxytest_images/2.jpg", char_seg[1]);
    cv::imwrite("python/zxytest_images/3.jpg", char_seg[2]);
    cv::imwrite("python/zxytest_images/4.jpg", char_seg[3]);
    cv::imwrite("python/zxytest_images/5.jpg", char_seg[4]);
    cv::imwrite("python/zxytest_images/6.jpg", char_seg[5]);
    cv::imwrite("python/zxytest_images/7.jpg", char_seg[6]);
    cv::destroyAllWindows();
    return true;
}


void sobelOper(const Mat &in, Mat &out, int blurSize, int morphW, int morphH) {
    Mat mat_blur;
    mat_blur = in.clone();
    GaussianBlur(in, mat_blur, Size(blurSize, blurSize), 0, 0, BORDER_DEFAULT);

    Mat mat_gray;
    if (mat_blur.channels() == 3)
        cvtColor(mat_blur, mat_gray, CV_RGB2GRAY);
    else
        mat_gray = mat_blur;

    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;


    Sobel(mat_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);

    Mat grad;
    addWeighted(abs_grad_x, 1, 0, 0, 0, grad);

    Mat mat_threshold;
    double otsu_thresh_val =
            threshold(grad, mat_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);


    Mat element = getStructuringElement(MORPH_RECT, Size(morphW, morphH));
    morphologyEx(mat_threshold, mat_threshold, MORPH_CLOSE, element);

    out = mat_threshold;
}


bool solve3() {
    char filename[200];
    sprintf(filename, "saveImage/1.jpg");
    Mat input_img = imread(filename, 1);
    Mat input_img2;
    input_img2 = input_img.clone();
    if (input_img.empty()) {
        return false;
    }
    vector<RotatedRect> rects;
    sobelOper(input_img, input_img, 3, 17, 3);
    //寻找车牌区域的轮廓
    vector<vector<Point> > contours;
    findContours(input_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); //只检测外面的轮廓
    //对候选的轮廓进行进一步的筛选
    vector<vector<Point> >::iterator it = contours.begin();
    while (it != contours.end()) {
        RotatedRect temp = minAreaRect(Mat(*it)); //返回每个轮廓的最小有界矩形区域
        if (!check_closeImage(temp)) {
            it = contours.erase(it);
        } else {
            rects.push_back(temp);
            ++it;
        }
    }
    Mat result;
    input_img2.copyTo(result);
    drawContours(result, contours, -1, Scalar(0, 0, 255), 1);
    vector<Mat> output_area;
    GetStandardPlate(input_img2, rects, output_area); //获得144×33的候选车牌区域
    CvSVM svmClassifyModel;
    svm_train(svmClassifyModel);
    vector<Mat> plates_svm;
    for (int i = 0; i < output_area.size(); i++) {
        Mat img = output_area[i];
        Mat now = img.reshape(1, 1);
        now.convertTo(now, CV_32FC1);
        int res = (int) svmClassifyModel.predict(now);
        if (res == 1)
            plates_svm.push_back(output_area[i]);
    }
    if (plates_svm.size() == 0) {
        return false;
    }
    vector<Mat> char_seg;
    global_flag = 1;
    char_segment(plates_svm[0], char_seg);
    if (global_flag == 0) return false;
    cv::imshow("char0", char_seg[0]);
    waitKey(0);
    cv::imshow("char1", char_seg[1]);
    waitKey(0);
    cv::imshow("char2", char_seg[2]);
    waitKey(0);
    cv::imshow("char3", char_seg[3]);
    waitKey(0);
    cv::imshow("char4", char_seg[4]);
    waitKey(0);
    cv::imshow("char5", char_seg[5]);
    waitKey(0);
    cv::imshow("char6", char_seg[6]);
    waitKey(0);
    cv::imwrite("python/zxytest_images/1.jpg", char_seg[0]);
    cv::imwrite("python/zxytest_images/2.jpg", char_seg[1]);
    cv::imwrite("python/zxytest_images/3.jpg", char_seg[2]);
    cv::imwrite("python/zxytest_images/4.jpg", char_seg[3]);
    cv::imwrite("python/zxytest_images/5.jpg", char_seg[4]);
    cv::imwrite("python/zxytest_images/6.jpg", char_seg[5]);
    cv::imwrite("python/zxytest_images/7.jpg", char_seg[6]);
    cv::destroyAllWindows();
    return true;
}

