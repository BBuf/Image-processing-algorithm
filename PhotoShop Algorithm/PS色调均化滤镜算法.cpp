Mat Equalize(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	int HistGram[256] = {0};
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			for (int k = 0; k < 3; k++) {
				HistGram[(int)src.at<Vec3b>(i, j)[k]]++;
			}
		}
	}
	int Num = 0;
	int LUT[256];
	for (int i = 0; i < 256; i++) {
		Num += HistGram[i];
		LUT[i] = (int)((float)Num / (row * col * 3) * 255);
	}
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			for (int k = 0; k < 3; k++) {
				dst.at<Vec3b>(i, j)[k] = LUT[(int)src.at<Vec3b>(i, j)[k]];
			}
		}
	}
	return dst;
}