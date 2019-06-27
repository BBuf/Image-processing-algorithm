//灰度图
const float PI = acos(-1.0);

Mat Emboss(Mat src, float angle=30, int offset=127) {
	int row = src.rows;
	int col = src.cols;
	float radian = angle * PI / 180.0;
	//float kernel[] =   // 灰色浮雕模板系数
	//{
	//	cos(radian + PI / 4), cos(radian + PI / 2), cos(radian + 3.0*PI / 4.0),
	//	cos(radian),        0,                  cos(PI),
	//	cos(radian - PI / 2), cos(radian - PI / 2), cos(radian - 3.0*PI / 4.0)
	//};
	float kernel[] =   // 彩色浮雕模板系数
	{
		cos(radian + PI / 4), cos(radian + PI / 2), cos(radian + 3.0*PI / 4.0),
			cos(radian), 1, cos(PI),
			cos(radian - PI / 2), cos(radian - PI / 2), cos(radian - 3.0*PI / 4.0)
	};
	Mat dst(row, col, CV_8UC1);
	int border = 1;
	for (int i = border; i < row - border; i++) {
		for (int j = border; j < col - border; j++) {
			float sum = 0;
			int id = 0;
			for (int x = -border; x <= border; x++) {
				for (int y = -border; y <= border; y++) {
					sum = sum + float(1.0 * src.at<uchar>(i + x, j + y) * kernel[id++]);
				}
			}
			if (sum > 255) sum = 255;
			else if (sum < 0) sum = 0;
			dst.at<uchar>(i, j) = sum;
		}
	}
	return dst;
}