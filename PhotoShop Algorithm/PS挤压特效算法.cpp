//degree代表挤压幅度，在1-32之间
Mat Pince(Mat src, int degree) {
	int row = src.rows;
	int col = src.cols;
	if (degree < 1) degree = 1;
	if (degree > 32) degree = 32;
	Mat dst(row, col, CV_8UC3);
	//图片的中心点
	int midX = col / 2;
	int midY = row / 2;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int offsetX = j - midX;
			int offsetY = i - midY;
			double radian = atan2((double)offsetY, (double)offsetX);
			double radius = sqrtf((double)(offsetX * offsetX + offsetY * offsetY));
			//实现挤压
			radius = sqrtf(radius) * degree;
			int X = (int)(radius * cos(radian)) + midX;
			int Y = (int)(radius * sin(radian)) + midY;
			if (X < 0) X = 0;
			if (X >= col) X = col - 1;
			if (Y < 0) Y = 0;
			if (Y >= row) Y = row - 1;
			for (int k = 0; k < 3; k++) {
				dst.at<Vec3b>(i, j)[k] = src.at<Vec3b>(Y, X)[k];
			}
		}
	}
	return dst;
}