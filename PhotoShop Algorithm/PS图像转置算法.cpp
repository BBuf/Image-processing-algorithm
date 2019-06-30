Mat Transposition(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(col, row, CV_8UC3);
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++) {
			for (int k = 0; k < 3; k++) {
				dst.at<Vec3b>(j, i) = src.at<Vec3b>(i, j);
			}
		}
	}
	return dst;
}
