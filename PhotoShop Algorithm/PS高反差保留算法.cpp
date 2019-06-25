Mat HighPass(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat gau(row, col, CV_8UC3);
	GaussianBlur(src, gau, Size(3, 3), 0, 0);
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			for (int k = 0; k < 3; k++) {
				dst.at<Vec3b>(i, j)[k] = src.at<Vec3b>(i, j)[k] - gau.at<Vec3b>(i, j)[k] + 127;
				if (dst.at<Vec3b>(i, j)[k] > 255)
					dst.at<Vec3b>(i, j)[k] = 255;
				else if (dst.at<Vec3b>(i, j)[k] < 0)
					dst.at<Vec3b>(i, j)[k] = 0;
			}
		}
	}
	return dst;
}