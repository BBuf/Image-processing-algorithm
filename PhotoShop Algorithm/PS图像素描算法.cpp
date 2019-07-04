Mat Sketch(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC1);
	Mat gray(row, col, CV_8UC1);
	cvtColor(src, dst, COLOR_BGR2GRAY);
	gray = dst.clone();
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			dst.at<uchar>(i, j) = 255 - dst.at<uchar>(i, j);
		}
	}
	GaussianBlur(dst, dst, Size(3, 3), 1.0);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			double b = double(dst.at<uchar>(i, j));
			double a = double(gray.at<uchar>(i, j));
			int temp = (int)(a + a*b / (256.0 - b));
			if (temp > 255)
				temp = 255;
			dst.at<uchar>(i, j) = temp;
		}
	}
	return dst;
}