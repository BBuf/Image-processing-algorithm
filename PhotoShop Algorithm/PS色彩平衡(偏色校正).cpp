Mat ColorBalance(Mat src, int cR, int cG, int cB) {
	int row = src.rows;
	int col = src.cols;
	//验证参数范围
	if (cR < -255)
		cR = -255;
	if (cR > 255)
		cR = 255;
	if (cG < -255)
		cG = -255;
	if (cG > 255)
		cG = 255;
	if (cB < -255)
		cB = -255;
	if (cB > 255)
		cB = 255;
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int B = src.at<Vec3b>(i, j)[0] + cB;
			int G = src.at<Vec3b>(i, j)[1] + cG;
			int R = src.at<Vec3b>(i, j)[2] + cR;
			if (B < 0)
				B = 0;
			else if (B > 255)
				B = 255;
			if (G < 0)
				G = 0;
			else if (G > 255)
				G = 255;
			if (R < 0)
				R = 0;
			else if (R > 255)
				R = 255;
			dst.at<Vec3b>(i, j)[0] = B;
			dst.at<Vec3b>(i, j)[1] = G;
			dst.at<Vec3b>(i, j)[2] = R;
		}
	}
	return dst;
}