Mat Posterize(Mat src, int Num) {
	int row = src.rows;
	int col = src.cols;
	int Step = 255 / (Num - 1);
	vector <int> T;
	for (int i = 1; i <= Num; i++) {
		T.push_back((i - 1) * Step);
	}
	Mat dst(row, col, CV_8UC3);
	Step = floor(255 / Num) + 1;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			for (int k = 0; k < 3; k++) {
				dst.at<Vec3b>(i, j)[k] = T[floor(src.at<Vec3b>(i, j)[k] / Step)];
			}
		}
	}
	return dst;
}
