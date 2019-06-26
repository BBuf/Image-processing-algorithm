Mat PaperCutArtFilter(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	float m = mean(gray)[0];
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (gray.at<uchar>(i, j) > m) {
				// 背景
				dst.at<Vec3b>(i, j)[0] = 0;
				dst.at<Vec3b>(i, j)[1] = 0;
				dst.at<Vec3b>(i, j)[2] = 139;
			}
			else {
				// 前景
				dst.at<Vec3b>(i, j)[0] = 255;
				dst.at<Vec3b>(i, j)[1] = 255;
				dst.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}
	return dst;
}