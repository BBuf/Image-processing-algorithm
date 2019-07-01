//flag 0:Mul 1:Div
Mat MulDiv(Mat src1, Mat src2, int flag) {
	int row = src1.rows;
	int col = src1.cols;
	if (row != src2.rows || col != src2.cols) {
		fprintf_s(stderr, "Input shape don't match!");
		exit(-1);
	}
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			for (int k = 0; k < 3; k++) {
				if(flag == 0){
					dst.at<Vec3b>(i, j)[k] = src1.at<Vec3b>(i, j)[k] * src2.at<Vec3b>(i, j)[k] / 255;
				}
				else {
					int val = (int)((1.0* src1.at<Vec3b>(i, j)[k] / (1.0 * src2.at<Vec3b>(i, j)[k] + 1.0)) * 255.0 + 0.5);
					if (val < 0) val = 0;
					else if (val > 255) val = 255;
					dst.at<Vec3b>(i, j)[k] = val;
				}
			}
		}
	}
	return dst;
}