const float YDbDrYRF = 0.299F;              // RGB转YDbDr的系数(浮点类型）
const float YDbDrYGF = 0.587F;
const float YDbDrYBF = 0.114F;
const float YDbDrDbRF = -0.1688F;
const float YDbDrDbGF = -0.3312F;
const float YDbDrDbBF = 0.5F;
const float YDbDrDrRF = -0.5F;
const float YDbDrDrGF = 0.4186F;
const float YDbDrDrBF = 0.0814F;

const float RGBRYF = 1.00000F;            // YDbDr转RGB的系数(浮点类型）
const float RGBRDbF = 0.0002460817072494899F;
const float RGBRDrF = -1.402083073344533F;
const float RGBGYF = 1.00000F;
const float RGBGDbF = -0.344268308442098F;
const float RGBGDrF = 0.714219609001458F;
const float RGBBYF = 1.00000F;
const float RGBBDbF = 1.772034373903893F;
const float RGBBDrF = 0.0002111539810593343F;

const int Shift = 20;
const int HalfShiftValue = 1 << (Shift - 1);

const int YDbDrYRI = (int)(YDbDrYRF * (1 << Shift) + 0.5);         // RGB转YDbDr的系数(整数类型）
const int YDbDrYGI = (int)(YDbDrYGF * (1 << Shift) + 0.5);
const int YDbDrYBI = (int)(YDbDrYBF * (1 << Shift) + 0.5);
const int YDbDrDbRI = (int)(YDbDrDbRF * (1 << Shift) + 0.5);
const int YDbDrDbGI = (int)(YDbDrDbGF * (1 << Shift) + 0.5);
const int YDbDrDbBI = (int)(YDbDrDbBF * (1 << Shift) + 0.5);
const int YDbDrDrRI = (int)(YDbDrDrRF * (1 << Shift) + 0.5);
const int YDbDrDrGI = (int)(YDbDrDrGF * (1 << Shift) + 0.5);
const int YDbDrDrBI = (int)(YDbDrDrBF * (1 << Shift) + 0.5);

const int RGBRYI = (int)(RGBRYF * (1 << Shift) + 0.5);              // YDbDr转RGB的系数(整数类型）
const int RGBRDbI = (int)(RGBRDbF * (1 << Shift) + 0.5);
const int RGBRDrI = (int)(RGBRDrF * (1 << Shift) + 0.5);
const int RGBGYI = (int)(RGBGYF * (1 << Shift) + 0.5);
const int RGBGDbI = (int)(RGBGDbF * (1 << Shift) + 0.5);
const int RGBGDrI = (int)(RGBGDrF * (1 << Shift) + 0.5);
const int RGBBYI = (int)(RGBBYF * (1 << Shift) + 0.5);
const int RGBBDbI = (int)(RGBBDbF * (1 << Shift) + 0.5);
const int RGBBDrI = (int)(RGBBDrF * (1 << Shift) + 0.5);

Mat RGB2YDbDr(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Blue = src.at<Vec3b>(i, j)[0];
			int Green = src.at<Vec3b>(i, j)[1];
			int Red = src.at<Vec3b>(i, j)[2];
			dst.at<Vec3b>(i, j)[0] = (uchar)((YDbDrYRI * Red + YDbDrYGI * Green + YDbDrYBI * Blue + HalfShiftValue) >> Shift);
			dst.at<Vec3b>(i, j)[1] = (uchar)(128 + ((YDbDrDbRI * Red + YDbDrDbGI * Green + YDbDrDbBI * Blue + HalfShiftValue) >> Shift));
			dst.at<Vec3b>(i, j)[2] = (uchar)(128 + ((YDbDrDrRI * Red + YDbDrDrGI * Green + YDbDrDrBI * Blue + HalfShiftValue) >> Shift));
		}
	}
	return dst;
}

Mat YDbDr2RGB(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Y = src.at<Vec3b>(i, j)[0];
			int Db = src.at<Vec3b>(i, j)[1] - 128;
			int Dr = src.at<Vec3b>(i, j)[2] - 128;
			int Red = Y + ((RGBRDbI * Db + RGBRDrI * Dr + HalfShiftValue) >> Shift);
			int Green = Y + ((RGBGDbI * Db + RGBGDrI * Dr + HalfShiftValue) >> Shift);
			int Blue = Y + ((RGBBDbI * Db + RGBBDrI * Dr + HalfShiftValue) >> Shift);
			if (Red > 255) Red = 255;
			else if (Red < 0) Red = 0;
			if (Green > 255) Green = 255;
			else if (Green < 0) Green = 0;
			if (Blue > 255) Blue = 0;
			else if (Blue < 0) Blue = 0;
			dst.at<Vec3b>(i, j)[0] = Blue;
			dst.at<Vec3b>(i, j)[1] = Green;
			dst.at<Vec3b>(i, j)[2] = Red;
		}
	}
	return dst;
}