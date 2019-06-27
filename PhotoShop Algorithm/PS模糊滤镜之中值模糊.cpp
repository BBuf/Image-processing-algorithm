Mat MedianFilter(Mat src, int ksize) {
	int row = src.rows;
	int col = src.cols;
	int border = (ksize - 1) / 2;
	int mid = (ksize*ksize - 1) / 2;
	Mat dst(row, col, CV_8UC3);
	for (int i = border; i < row - border; i++) {
		for (int j = border; j < col - border; j++) {
			for (int k = 0; k < 3; k++) {
				vector <int> v;
				for (int x = -border; x <= border; x++) {
					for (int y = -border; y <= border; y++) {
						v.push_back(src.at<Vec3b>(i + x, j + y)[k]);
					}
				}
				sort(v.begin(), v.end());
				dst.at<Vec3b>(i, j)[k] = v[mid];
			}
		}
	}
	return dst;
}