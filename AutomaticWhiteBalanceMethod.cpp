const float YCbCrYRF = 0.299F;              // RGB转YCbCr的系数(浮点类型）
const float YCbCrYGF = 0.587F;
const float YCbCrYBF = 0.114F;
const float YCbCrCbRF = -0.168736F;
const float YCbCrCbGF = -0.331264F;
const float YCbCrCbBF = 0.500000F;
const float YCbCrCrRF = 0.500000F;
const float YCbCrCrGF = -0.418688F;
const float YCbCrCrBF = -0.081312F;

const float RGBRYF = 1.00000F;            // YCbCr转RGB的系数(浮点类型）
const float RGBRCbF = 0.0000F;
const float RGBRCrF = 1.40200F;
const float RGBGYF = 1.00000F;
const float RGBGCbF = -0.34414F;
const float RGBGCrF = -0.71414F;
const float RGBBYF = 1.00000F;
const float RGBBCbF = 1.77200F;
const float RGBBCrF = 0.00000F;

const int Shift = 20;
const int HalfShiftValue = 1 << (Shift - 1);

const int YCbCrYRI = (int)(YCbCrYRF * (1 << Shift) + 0.5);         // RGB转YCbCr的系数(整数类型）
const int YCbCrYGI = (int)(YCbCrYGF * (1 << Shift) + 0.5);
const int YCbCrYBI = (int)(YCbCrYBF * (1 << Shift) + 0.5);
const int YCbCrCbRI = (int)(YCbCrCbRF * (1 << Shift) + 0.5);
const int YCbCrCbGI = (int)(YCbCrCbGF * (1 << Shift) + 0.5);
const int YCbCrCbBI = (int)(YCbCrCbBF * (1 << Shift) + 0.5);
const int YCbCrCrRI = (int)(YCbCrCrRF * (1 << Shift) + 0.5);
const int YCbCrCrGI = (int)(YCbCrCrGF * (1 << Shift) + 0.5);
const int YCbCrCrBI = (int)(YCbCrCrBF * (1 << Shift) + 0.5);

const int RGBRYI = (int)(RGBRYF * (1 << Shift) + 0.5);              // YCbCr转RGB的系数(整数类型）
const int RGBRCbI = (int)(RGBRCbF * (1 << Shift) + 0.5);
const int RGBRCrI = (int)(RGBRCrF * (1 << Shift) + 0.5);
const int RGBGYI = (int)(RGBGYF * (1 << Shift) + 0.5);
const int RGBGCbI = (int)(RGBGCbF * (1 << Shift) + 0.5);
const int RGBGCrI = (int)(RGBGCrF * (1 << Shift) + 0.5);
const int RGBBYI = (int)(RGBBYF * (1 << Shift) + 0.5);
const int RGBBCbI = (int)(RGBBCbF * (1 << Shift) + 0.5);
const int RGBBCrI = (int)(RGBBCrF * (1 << Shift) + 0.5);

Mat RGB2YCbCr(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Blue = src.at<Vec3b>(i, j)[0];
			int Green = src.at<Vec3b>(i, j)[1];
			int Red = src.at<Vec3b>(i, j)[2];
			dst.at<Vec3b>(i, j)[0] = (int)((YCbCrYRI * Red + YCbCrYGI * Green + YCbCrYBI * Blue + HalfShiftValue) >> Shift);
			dst.at<Vec3b>(i, j)[1] = (int)(128 + ((YCbCrCbRI * Red + YCbCrCbGI * Green + YCbCrCbBI * Blue + HalfShiftValue) >> Shift));
			dst.at<Vec3b>(i, j)[2] = (int)(128 + ((YCbCrCrRI * Red + YCbCrCrGI * Green + YCbCrCrBI * Blue + HalfShiftValue) >> Shift));
		}
	}
	return dst;
}

Mat YCbCr2RGB(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Y = src.at<Vec3b>(i, j)[0];
			int Cb = src.at<Vec3b>(i, j)[1] - 128;
			int Cr = src.at<Vec3b>(i, j)[2] - 128;
			int Red = Y + ((RGBRCrI * Cr + HalfShiftValue) >> Shift);
			int Green = Y + ((RGBGCbI * Cb + RGBGCrI * Cr + HalfShiftValue) >> Shift);
			int Blue = Y + ((RGBBCbI * Cb + HalfShiftValue) >> Shift);
			if (Red > 255) Red = 255; else if (Red < 0) Red = 0;
			if (Green > 255) Green = 255; else if (Green < 0) Green = 0;    // 编译后应该比三目运算符的效率高
			if (Blue > 255) Blue = 255; else if (Blue < 0) Blue = 0;
			dst.at<Vec3b>(i, j)[0] = Blue;
			dst.at<Vec3b>(i, j)[1] = Green;
			dst.at<Vec3b>(i, j)[2] = Red;
		}
	}
	return dst;
}

Mat AutomaticWhiteBalanceMethod(Mat src) {
	int row = src.rows;
	int col = src.cols;
	if (src.channels() == 4) {
		cvtColor(src, src, CV_BGRA2BGR);
	}
	Mat input = RGB2YCbCr(src);
	Mat mark(row, col, CV_8UC1);
	int sum = 0;
	for (int i = 0; i < row; i+=100) {
		for (int j = 0; j < col; j+=100) {
			if (i + 100 < row && j + 100 < col) {
				Rect rect(j, i, 100, 100);
				Mat temp = input(rect);
				Scalar global_mean = mean(temp);
				double dr = 0, db = 0;
				for (int x = 0; x < 100; x++) {
					uchar *ptr = temp.ptr<uchar>(x) + 1;
					for (int y = 0; y < 100; y++) {
						dr += pow(abs(*ptr - global_mean[1]), 2);
						ptr++;
						db += pow(abs(*ptr - global_mean[2]), 2);
						ptr++;
						ptr++;
					}
				}
				dr /= 10000;
				db /= 10000;
				double cr_left_criteria = 1.5 * global_mean[1] + dr * global_mean[1];
				double cr_right_criteria = 1.5 * dr;
				double cb_left_criteria = global_mean[2] + db * global_mean[2];
				double cb_right_criteria = 1.5 * db;
				for (int x = 0; x < 100; x++) {
					uchar *ptr = temp.ptr<uchar>(x) + 1;
					for (int y = 0; y < 100; y++) {
						uchar cr = *ptr;
						ptr++;
						uchar cb = *ptr;
						ptr++;
						ptr++;
						if ((cr - cb_left_criteria) < cb_right_criteria && (cb - cr_left_criteria) < cr_right_criteria) {
							sum++;
							mark.at<uchar>(i + x, j + y) = 1;
						}
						else {
							mark.at<uchar>(i + x, j + y) = 0;
						}
					}
				}
			}
		}
	}

	int Threshold = 0;
	int Ymax = 0;
	int Light[256] = { 0 };
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (mark.at<uchar>(i, j) == 1) {
				Light[(int)(input.at<Vec3b>(i, j)[0])]++;
			}
			Ymax = max(Ymax, (int)(input.at<Vec3b>(i, j)[0]));
		}
	}
	printf("maxY: %d\n", Ymax);
	int sum2 = 0;
	for (int i = 255; i >= 0; i--) {
		sum2 += Light[i];
		if (sum2 >= sum * 0.1) {
			Threshold = i;
			break;
		}
	}
	printf("Threshold: %d\n", Threshold);
	printf("Sum: %d Sum2: %d\n", sum, sum2);
	double Blue = 0;
	double Green = 0;
	double Red = 0;
	int cnt2 = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (mark.at<uchar>(i, j) == 1 && (int)(input.at<Vec3b>(i, j)[0]) >= Threshold) {
				Blue += 1.0 * src.at<Vec3b>(i, j)[0];
				Green += 1.0 * src.at<Vec3b>(i, j)[1];
				Red += 1.0 * src.at<Vec3b>(i, j)[2];
				cnt2++;
			}
		}
	}
	Blue /= cnt2;
	Green /= cnt2;
	Red /= cnt2;
	printf("%.5f %.5f %.5f\n", Blue, Green, Red);
	Mat dst(row, col, CV_8UC3);
	double maxY = Ymax;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int B = (int)(maxY * src.at<Vec3b>(i, j)[0] / Blue);
			int G = (int)(maxY * src.at<Vec3b>(i, j)[1] / Green);
			int R = (int)(maxY * src.at<Vec3b>(i, j)[2] / Red);
			if (B > 255) B = 255; else if (B < 0) B = 0;
			if (G > 255) G = 255; else if (G < 0) G = 0;
			if (R > 255) R = 255; else if (R < 0) R = 0;
			dst.at<Vec3b>(i, j)[0] = B;
			dst.at<Vec3b>(i, j)[1] = G;
			dst.at<Vec3b>(i, j)[2] = R;
		}
	}
	return dst;
}