Mat Bright(Mat src, int cb_) {
	int row = src.rows;
	int col = src.cols;
	if (cb_ < -255)
		cb_ = -255;
	if (cb_ > 255)
		cb_ = 255;
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			for (int k = 0; k < 3; k++) {
				int val = src.at<Vec3b>(i, j)[k] + cb_;
				if (val < 0) val = 0;
				else if (val > 255) val = 255;
				dst.at<Vec3b>(i, j)[k] = val;
			}
		}
	}
	return dst;
}