//flag 0:Add 1:Sub
Mat AddSub(Mat src1, Mat src2, int flag) {
	int row = src1.rows;
	int col = src1.cols;
	if (row != src2.rows || col != src2.cols) {
		fprintf_s(stderr, "Input shape don't match!");
		exit(-1);
	}
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			for (int k = 0; k < 3; k++) {
				if (flag == 0) {
					int val = src1.at<Vec3b>(i, j)[k] + src2.at<Vec3b>(i, j)[k];
					if (val > 255) val = 255;
					dst.at<Vec3b>(i, j)[k] = val;
				}
				else {
					int val = src1.at<Vec3b>(i, j)[k] - src2.at<Vec3b>(i, j)[k];
					if (val < 0) val = 0;
					dst.at<Vec3b>(i, j)[k] = val;
				}
			}
		}
	}
	return dst;
}
