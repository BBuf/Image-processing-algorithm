// degree:钝化度，取值（0~100）
// 钝化度用来改变像素间的对比度强弱，钝化值越小，钝化的部分就越窄，仅仅会影响边缘像素
// 钝化值越大，钝化的范围越宽，效果更明显
Mat UnsharpMask(Mat src, int degree) {
	int row = src.rows;
	int col = src.cols;
	if (degree < 1) degree = 1;
	if (degree > 100) degree = 100;
	Mat dst(row, col, CV_8UC3);
	src.copyTo(src);
	int border = 1;
	for (int i = 0; i < degree; i++) {
		GaussianBlur(dst, dst, Size(3, 3), 1.0);
	}
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			for (int k = 0; k < 3; k++) {
				int sum = 2 * src.at<Vec3b>(i, j)[k] - dst.at<Vec3b>(i, j)[k];
				if (sum > 255) sum = 255;
				else if (sum < 0) sum = 0;
				dst.at<Vec3b>(i, j)[k] = sum;
			}
		}
	}
	return dst;
}