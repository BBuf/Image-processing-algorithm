Mat TemplateBlur(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	int border = 1;
	for (int i = border; i < row - border; i++) {
		for (int j = border; j < col - border; j++) {
			for (int k = 0; k < 3; k++) {
				int sum = src.at<Vec3b>(i, j)[k] + src.at<Vec3b>(i + 1, j)[k] + src.at<Vec3b>(i - 1, j)[k] +
					src.at<Vec3b>(i, j - 1)[k] + src.at<Vec3b>(i + 1, j - 1)[k] + src.at<Vec3b>(i - 1, j - 1)[k] +
					src.at<Vec3b>(i, j + 1)[k] + src.at<Vec3b>(i + 1, j + 1)[k] + src.at<Vec3b>(i - 1, j + 1)[k];
				sum /= 9;
				dst.at<Vec3b>(i, j)[k] = sum;
			}
		}
	}
	return dst;
}