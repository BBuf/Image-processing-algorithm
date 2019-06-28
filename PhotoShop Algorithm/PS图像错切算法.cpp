//图像错切
//flag=1，水平错切; flag=-1, 垂直错切
//水平方向: x2=(x1-y1*tan(theta))
//         y2=y1
const double PI = 3.1415926;
Mat Slant(const Mat &src, float angle, int flag) {
	int rows = src.rows;
	int cols = src.cols;
	float ftan = fabs((float)tan(angle / 180.0*PI));
	int newHeight = 0;
	int newWidth = 0;
	if (flag == 1) { //水平方向高度不变
		newHeight = rows;
		newWidth = (int)(cols + rows*fabs(ftan));
	}
	else {//垂直方向宽度不变
		newHeight = (int)(rows + cols*fabs(ftan));
		newWidth = cols;
	}
	Mat dst(rows, cols, CV_8UC3);
	for (int i = 0; i < newHeight; i++) {
		for (int j = 0; j < newWidth; j++) {
			int newi, newj;
			if (flag == 1) {
				newi = i;
				newj = j + ftan * (i - rows);
			}
			else {
				newi = i + ftan * (j - cols);
				newj = j;
			}
			if (newi >= 0 && newi < rows && newj >= 0 && newj < cols) {
				for (int k = 0; k < 3; k++) {
					dst.at<Vec3b>(i, j)[k] = src.at<Vec3b>(newi, newj)[k];
				}
			}
			else {
				for (int k = 0; k < 3; k++) {
					dst.at<Vec3b>(i, j)[k] = 255;
				}
			}
		}
	}
	return dst;
}