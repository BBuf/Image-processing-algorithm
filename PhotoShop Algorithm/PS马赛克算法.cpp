Mat Mosaic(Mat src, int Ksize) {
	int offset = (Ksize - 1) / 2;
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	for (int i = offset; i < row - offset; i += 2 * offset) {
		for (int j = offset; j < col - offset; j += 2 * offset) {
			int b = src.at<Vec3b>(i, j)[0];
			int g = src.at<Vec3b>(i, j)[1];
			int r = src.at<Vec3b>(i, j)[2];
			for (int x = -offset; x < offset; x++) {
				for (int y = -offset; y < offset; y++) {
					dst.at<Vec3b>(i + x, j + y)[0] = b;
					dst.at<Vec3b>(i + x, j + y)[1] = g;
					dst.at<Vec3b>(i + x, j + y)[2] = r;
				}
			}
		}
	}
	return dst;
}
