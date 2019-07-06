Mat GaussDistributionMask(Mat src) {
	int row = src.rows;
	int col = src.cols;
	int center_x = row / 2;
	int center_y = col / 2;
	//获得二维高斯分布蒙版
	int R = sqrt(center_x * center_x + center_y * center_y);
	Mat Gauss_map(row, col, CV_32FC1);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float dis = sqrt(1.0 * (i - center_x) * (i - center_x) + 1.0 * (j - center_y) * (j - center_y));
			Gauss_map.at<float>(i, j) = exp(-0.5 * dis / R);
		}
	}
	//和原图相乘得到渐变映射的效果
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			for (int k = 0; k < 3; k++) {
				dst.at<Vec3b>(i, j)[k] = src.at<Vec3b>(i, j)[k] * Gauss_map.at<float>(i, j);
			}
		}
	}
	return dst;
}