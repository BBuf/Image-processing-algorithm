//径向模糊：缩放
//num: 均值力度
Mat RadialBlurZoom(Mat src, int num=10) {
	int row = src.rows;
	int col = src.cols;
	Point center(row / 2, col / 2);
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int R = norm(Point(i, j) - center);
			float angle = atan2(float(i - center.x), float(j - center.y));
			int sum1 = 0, sum2 = 0, sum3 = 0;
			for (int k = 0; k < num; k++) {
				int tmpR = (R - k) > 0 ? (R - k) : 0;
				int newX = tmpR * sin(angle) + center.x;
				int newY = tmpR * cos(angle) + center.y;
				if (newX < 0) newX = 0;
				if (newX > row - 1) newX = row - 1;
				if (newY < 0) newY = 0;
				if (newY > col - 1) newY = col - 1;
				sum1 += src.at<Vec3b>(newX, newY)[0];
				sum2 += src.at<Vec3b>(newX, newY)[1];
				sum3 += src.at<Vec3b>(newX, newY)[2];
			}
			dst.at<Vec3b>(i, j)[0] = (uchar)(sum1 / num);
			dst.at<Vec3b>(i, j)[1] = (uchar)(sum2 / num);
			dst.at<Vec3b>(i, j)[2] = (uchar)(sum3 / num);
		}
	}
	return dst;
}

//径向模糊：旋转
Mat RadialBlurRotate(Mat src, int num = 10) {
	int row = src.rows;
	int col = src.cols;
	Point center(row / 2, col / 2);
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int R = norm(Point(i, j) - center);
			float angle = atan2(float(i - center.x), float(j - center.y));
			int sum1 = 0, sum2 = 0, sum3 = 0;
			for (int k = 0; k < num; k++) {
				angle += 0.01;
				int newX = R * sin(angle) + center.x;
				int newY = R * cos(angle) + center.y;
				if (newX < 0) newX = 0;
				if (newX > row - 1) newX = row - 1;
				if (newY < 0) newY = 0;
				if (newY > col - 1) newY = col - 1;
				sum1 += src.at<Vec3b>(newX, newY)[0];
				sum2 += src.at<Vec3b>(newX, newY)[1];
				sum3 += src.at<Vec3b>(newX, newY)[2];
			}
			dst.at<Vec3b>(i, j)[0] = (uchar)(sum1 / num);
			dst.at<Vec3b>(i, j)[1] = (uchar)(sum2 / num);
			dst.at<Vec3b>(i, j)[2] = (uchar)(sum3 / num);
		}
	}
	return dst;
}
