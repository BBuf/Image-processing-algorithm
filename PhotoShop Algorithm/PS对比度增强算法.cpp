Mat Contrast(Mat src, int degree) {
	int row = src.rows;
	int col = src.cols;
	//验证参数范围
	if (degree < -100) degree = -100;
	if (degree > 100) degree = 100;
	double contrast = (100.0 + degree) / 100.0;
	contrast *= contrast;
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			for (int k = 0; k < 3; k++) {
				int val = (int)(((1.0 * src.at<Vec3b>(i, j)[k] / 255.0 - 0.5) * contrast + 0.5) * 255);
				if (val < 0) val = 0;
				else if (val > 255) val = 255;
				dst.at<Vec3b>(i, j)[k] = val;
			}
		}
	}
	return dst;
}
