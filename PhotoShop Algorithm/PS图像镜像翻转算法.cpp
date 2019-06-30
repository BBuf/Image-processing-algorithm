//flag=1水平镜像，flag=-1垂直镜像
Mat Mirror(Mat src, int flag) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	if (flag == 1) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				for (int k = 0; k < 3; k++) {
					dst.at<Vec3b>(i, j)[k] = src.at<Vec3b>(i, col - 1 - j)[k];
				}
			}
		}
	}
	else {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				for (int k = 0; k < 3; k++) {
					dst.at<Vec3b>(i, j)[k] = src.at<Vec3b>(row - 1 - i, j)[k];
				}
			}
		}
	}
	return dst;
}