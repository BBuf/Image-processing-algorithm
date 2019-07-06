Mat ColorClot(Mat src, int radius) {
	int row = src.rows;
	int col = src.cols;
	int border = (radius - 1) / 2;
	Mat dst(row, col, CV_8UC3);
	for (int i = border; i + border < row; i++) {
		for (int j = border; j + border < col; j++) {
			for (int k = 0; k < 3; k++) {
				int val = src.at<Vec3b>(i, j)[k];
				for (int x = -border; x <= border; x++) {
					for (int y = -border; y <= border; y++) {
						val = min(val, (int)src.at<Vec3b>(i + x, j + y)[k]);
					}
				}
				dst.at<Vec3b>(i, j)[k] = val;
			}
		}
	}
	return dst;
}