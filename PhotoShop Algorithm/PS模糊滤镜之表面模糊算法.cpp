Mat SurfaceBlur(Mat src, int radius) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	double Thres = 25;
	int border = (radius - 1) / 2;
	for (int k = 0; k < 3; k++) {
		for (int i = border; i + border < row; i++) {
			for (int j = border; j + border < col; j++) {
				double sum1 = 0;
				double sum2 = 0;
				for (int x = -border; x <= border; x++) {
					for (int y = -border; y <= border; y++) {
						double w = 1.0 - (abs(src.at<Vec3b>(i + x, j + y)[k] - src.at<Vec3b>(i, j)[k]) / (2.5 * Thres));
						if (w < 0) w = 0;
						sum1 += w * src.at<Vec3b>(i+x, j+y)[k];
						sum2 += w;
					}
				}
				dst.at<Vec3b>(i, j)[k] = sum1 / sum2;
			}
		}
	}
	return dst;
}