#include "opencv2/opencv.hpp"
#include "iostream"
#include "algorithm"
#include "vector"
using namespace std;
using namespace cv;

//计算大气光值
vector <int> m_anAirlight;
void AirlightEstimation(cv::Mat src)
{
    int nMinDistance = 65536;
    int nDistance;
    int nMaxIndex;
    double dpScore[3];
    float afScore[4] = {0};
    float nMaxScore = 0;
    int cols = src.cols;
    int rows = src.rows;
    //4 sub-block
    Mat R = Mat(rows / 2, cols / 2, CV_8UC1);
    Mat G = Mat(rows / 2, cols / 2, CV_8UC1);
    Mat B = Mat(rows / 2, cols / 2, CV_8UC1);
    Rect temp1(0, 0, cols / 2, rows / 2);
    Mat UpperLeft = src(temp1);
    Rect temp2(cols / 2, 0, cols / 2, rows / 2);
    Mat UpperRight = src(temp2);
    Rect temp3(0, rows / 2, cols / 2, rows / 2);
    Mat LowerLeft = src(temp3);
    Rect temp4(cols / 2, rows / 2, cols / 2, rows / 2);
    Mat LowerRight = src(temp4);
    if(rows * cols > 200){
        vector <Mat> channels;
        //upper left sub-block
        split(UpperLeft, channels);

        B = channels[0];
        G = channels[1];
        R = channels[2];
        Mat tmp_m, tmp_std;
        meanStdDev(R, tmp_m, tmp_std);
        dpScore[0] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        meanStdDev(G, tmp_m, tmp_std);
        dpScore[1] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        meanStdDev(B, tmp_m, tmp_std);
        dpScore[2] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        afScore[0] = (float)(dpScore[0] + dpScore[1] + dpScore[2]);
        nMaxScore = afScore[0];
        nMaxIndex = 0;
        //upper right sub-block
        split(UpperRight, channels);
        B = channels[0];
        G = channels[1];
        R = channels[2];
        meanStdDev(R, tmp_m, tmp_std);
        dpScore[0] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        meanStdDev(G, tmp_m, tmp_std);
        dpScore[1] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        meanStdDev(B, tmp_m, tmp_std);
        dpScore[2] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        afScore[1] = (float)(dpScore[0] + dpScore[1] + dpScore[2]);
        if(afScore[1] > nMaxScore){
            nMaxScore = afScore[1];
            nMaxIndex = 1;
        }
        //lower left sub-block
        split(LowerLeft, channels);
        B = channels[0];
        G = channels[1];
        R = channels[2];
        meanStdDev(R, tmp_m, tmp_std);
        dpScore[0] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        meanStdDev(G, tmp_m, tmp_std);
        dpScore[1] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        meanStdDev(B, tmp_m, tmp_std);
        dpScore[2] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        afScore[2] = (float)(dpScore[0] + dpScore[1] + dpScore[2]);
        if(afScore[2] > nMaxScore){
            nMaxScore = afScore[2];
            nMaxIndex = 2;
        }
        //lower right sub-block
        split(LowerRight, channels);
        B = channels[0];
        G = channels[1];
        R = channels[2];
        meanStdDev(R, tmp_m, tmp_std);
        dpScore[0] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        meanStdDev(G, tmp_m, tmp_std);
        dpScore[1] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        meanStdDev(B, tmp_m, tmp_std);
        dpScore[2] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        afScore[3] = (float)(dpScore[0] + dpScore[1] + dpScore[2]);
        if(afScore[3] > nMaxScore){
            nMaxScore = afScore[3];
            nMaxIndex = 3;
        }
        //select the sub-block, which has maximum score

        switch (nMaxIndex){
            case 0:
                AirlightEstimation(UpperLeft); break;
            case 1:
                AirlightEstimation(UpperRight); break;
            case 2:
                AirlightEstimation(LowerLeft); break;
            case 3:
                AirlightEstimation(LowerRight); break;
        }
    }else{
        //在子快中寻找最亮的点作为A
        printf("%d %d\n", src.rows, src.cols);
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                nDistance = int(sqrt(float(255 - src.at<Vec3b>(i, j)[0]) * float(255 - src.at<Vec3b>(i, j)[0])) +
                        sqrt(float(255 - src.at<Vec3b>(i, j)[1]) * float(255 - src.at<Vec3b>(i, j)[1])) +
                                        sqrt(float(255 - src.at<Vec3b>(i, j)[2]) * float(255 - src.at<Vec3b>(i, j)[2])));
                if(nMinDistance > nDistance){
                    m_anAirlight.clear();
                    nMinDistance = nDistance;
                    m_anAirlight.push_back(src.at<Vec3b>(i, j)[0]);
                    m_anAirlight.push_back(src.at<Vec3b>(i, j)[1]);
                    m_anAirlight.push_back(src.at<Vec3b>(i, j)[2]);
                }
            }
        }
        printf("success\n");
    }

}

//计算透射率
float NFTrsEstimationColor(cv::Mat src, float lamda=5.0){
    int rows = src.rows;
    int cols = src.cols;
    int nOutR, nOutG, nOutB, nSquaredOut, nSumofOuts, nSumofSquaredOuts;
    float fTrans, fOptTrs;
    int nTrans, nSumofLoss;
    float fCost, fMinCost, fMean;
    int nNumberofPixels, nLossCount;
    fTrans = 0.4f;
    nTrans = 427;
    nNumberofPixels = rows * cols * 3;
    for(int cnt = 0; cnt < 5; cnt++){
        nSumofLoss = 0;
        nLossCount = 0;
        nSumofSquaredOuts = 0;
        nSumofOuts = 0;
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                nOutB = ((src.at<Vec3b>(i, j)[0] - m_anAirlight[0]) * nTrans + 128 * m_anAirlight[0]) >> 7; //(I-A)/t+A-->((I-A)*k*128+A*128)/128
                nOutG = ((src.at<Vec3b>(i, j)[1] - m_anAirlight[1]) * nTrans + 128 * m_anAirlight[1]) >> 7;
                nOutR = ((src.at<Vec3b>(i, j)[2] - m_anAirlight[2]) * nTrans + 128 * m_anAirlight[2]) >> 7;
                if(nOutR>255){
                    nSumofLoss += (nOutR-255)*(nOutR-255);
                    nLossCount++;
                }else if(nOutR<0){
                    nSumofLoss += nOutR*nOutR;
                    nLossCount++;
                }
                if(nOutG>255){
                    nSumofLoss += (nOutG-255)*(nOutG-255);
                    nLossCount++;
                }else if(nOutG<0){
                    nSumofLoss += nOutG*nOutG;
                    nLossCount++;
                }
                if(nOutB>255){
                    nSumofLoss += (nOutB-255)*(nOutB-255);
                    nLossCount++;
                }else if(nOutB<0){
                    nSumofLoss += nOutB*nOutB;
                    nLossCount++;
                }
                nSumofSquaredOuts += nOutB*nOutB + nOutR*nOutR + nOutG*nOutG;
                nSumofOuts += nOutB + nOutG + nOutR;
            }
        }
        fMean = (float)(nSumofOuts)/(float)(nNumberofPixels);
        fCost = lamda * (float)nSumofLoss / (float)(nNumberofPixels) - ((float)nSumofSquaredOuts/(float)nNumberofPixels-fMean*fMean);
        if(cnt == 0 || fMinCost > fCost){
            fMinCost = fCost;
            fOptTrs = fTrans;
        }
        fTrans += 0.1f;
        nTrans = (int)(1.0f/fTrans*128.0f);
    }
    return fOptTrs;
}


//float NFTrsEstimationColor(cv::Mat src){
//    float t = 0.0;
//    int rows = src.rows;
//    int cols = src.cols;
//    float mi = 65536.0;
//    float mx = 0.0;
//    for(int i = 0; i < rows; i++){
//        for(int j = 0; j < cols; j++){
//            for(int k = 0; k < 3; k++){
//                mi = std::min(mi, ((float)src.at<Vec3b>(i, j)[k] - (float)m_anAirlight[k]) / ((float)(-m_anAirlight[k])));
//                mx = std::max(mx, ((float)src.at<Vec3b>(i, j)[k] - (float)m_anAirlight[k]) / float(255.0-(float)m_anAirlight[k]));
//            }
//        }
//    }
//    printf("%.5f %.5f\n", mi, mx);
//    t = max(mi, mx);
//    return t;
//}

int main(){
    Mat src = cv::imread("./org-canon.png");
    m_anAirlight.clear();
    AirlightEstimation(src);
    printf("%d %d %d\n", m_anAirlight[0], m_anAirlight[1], m_anAirlight[2]);
    int rows = src.rows;
    int cols = src.cols;
    Mat dst(rows, cols, CV_8UC3);
    int m_nTVlockSize = 41;
    for(int nY = 0; nY+m_nTVlockSize < rows; nY+=m_nTVlockSize){
        for(int nX = 0; nX+m_nTVlockSize < cols; nX+=m_nTVlockSize){
             Rect temp(nX, nY, m_nTVlockSize, m_nTVlockSize);
             Mat now = src(temp);
             float t = NFTrsEstimationColor(now);
             //printf("%.3f\n", t);
             //float t = 0.;
             for(int i = 0; i < m_nTVlockSize; i++){
                 for(int j = 0; j < m_nTVlockSize; j++){
                     for(int k = 0; k < 3; k++){
                         dst.at<Vec3b>(nY+i, nX+j)[k] = int((double)(src.at<Vec3b>(nY+i, nX+j)[k] - m_anAirlight[k]) / t) + m_anAirlight[k];
                         if(dst.at<Vec3b>(nY+i, nX+j)[k] > 255) dst.at<Vec3b>(nY+i, nX+j)[k] = 255;
                         else if(dst.at<Vec3b>(nY+i, nX+j)[k] < 0) dst.at<Vec3b>(nY+i, nX+j)[k] = 0;
                     }
                 }
             }
        }
    }
    cv::imshow("origin", src);
    cv::imshow("result", dst);
    waitKey(0);
    return 0;
}