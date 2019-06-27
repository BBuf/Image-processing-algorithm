Mat LaplaceSharp(Mat src) {
	int row = src.rows;
	int col = src.cols;
	int border = 1;
	Mat dst(row, col, CV_8UC3);
	for (int i = border; i < row - border; i++) {
		for (int j = border; j < col - border; j++) {
			for (int k = 0; k < 3; k++) {
				int sum = 9 * src.at<Vec3b>(i, j)[k] - src.at<Vec3b>(i - 1, j - 1)[k] - src.at<Vec3b>(i - 1, j)[k]
					- src.at<Vec3b>(i - 1, j + 1)[k] - src.at<Vec3b>(i, j - 1)[k] - src.at<Vec3b>(i, j + 1)[k]
					- src.at<Vec3b>(i + 1, j - 1)[k] - src.at<Vec3b>(i + 1, j)[k] - src.at<Vec3b>(i + 1, j + 1)[k];
				if (sum > 255) sum = 255;
				else if (sum < 0) sum = 0;
				dst.at<Vec3b>(i, j)[k] = sum;
			}
		}
	}
	return dst;
}
