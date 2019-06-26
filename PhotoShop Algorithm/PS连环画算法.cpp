Mat ComicStrips(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int b = src.at<Vec3b>(i, j)[0];
			int g = src.at<Vec3b>(i, j)[1];
			int r = src.at<Vec3b>(i, j)[2];
			int R = abs(g - b + g + r) * r / 256;
			int G = abs(b - g + b + r) * r / 256;
			int B = abs(b - g + b + r) * g / 256;
			if (R > 255) R = 255;
			if (R < 0) R = 0;
			if (G > 255) G = 255;
			if (G < 0) G = 0;
			if (B > 255) B = 255;
			if (B < 0) B = 0;
			dst.at<Vec3b>(i, j)[0] = B;
			dst.at<Vec3b>(i, j)[1] = G;
			dst.at<Vec3b>(i, j)[2] = R;
		}
	}
	Mat gray;
	cvtColor(dst, gray, COLOR_BGR2GRAY);
	return gray;
}