Mat Equalizer(Mat src) {
	int row = src.rows;
	int col = src.cols;
	int Count[256] = { 0 };
	float p[256] = { 0 };
	float fSum[256] = { 0 };
	int level[256] = { 0 };
	int Total = row * col * 3;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			for (int k = 0; k < 3; k++) {
				Count[src.at<Vec3b>(i, j)[k]]++;
			}
		}
	}
	for (int i = 0; i < 256; i++) {
		p[i] = 1.0 * Count[i] / (1.0 * Total);
		if (i == 0)
			fSum[0] = p[0];
		else
			fSum[i] = fSum[i - 1] + p[i];
		level[i] = saturate_cast<uchar>(255 * fSum[i] + 0.5);
	}
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			for (int k = 0; k < 3; k++) {
				dst.at<Vec3b>(i, j)[k] = level[src.at<Vec3b>(i, j)[k]];
			}
		}
	}
	return dst;
}