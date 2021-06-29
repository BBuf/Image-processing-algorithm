//最邻近插值
Mat NearestNeighborInterpolation(Mat src, float sx, float sy) {
	int row = src.rows;
	int col = src.cols;
	int channels = src.channels();
	int dst_row = round(row * sx);
	int dst_col = round(col * sy);
	if (channels == 1) {
		Mat dst(dst_row, dst_col, CV_8UC1);
		for (int i = 0; i < dst_row; i++) {
			for (int j = 0; j < dst_col; j++) {
				int pre_i = floor(i / sy);
				int pre_j = floor(j / sx);
				if (pre_i > row - 1) pre_i = row - 1;
				if (pre_j > col - 1) pre_j = col - 1;
				dst.at<uchar>(i, j) = src.at<uchar>(pre_i, pre_j);
			}
		}
		return dst;
	}
	else {
		Mat dst(dst_row, dst_col, CV_8UC3);
		for (int i = 0; i < dst_row; i++) {
			for (int j = 0; j < dst_col; j++) {
				int pre_i = floor(i / sy);
				int pre_j = floor(j / sx);
				if (pre_i > row - 1) pre_i = row - 1;
				if (pre_j > col - 1) pre_j = col - 1;
				dst.at<Vec3b>(i, j)[0] = src.at<Vec3b>(pre_i, pre_j)[0];
				dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(pre_i, pre_j)[1];
				dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(pre_i, pre_j)[2];
			}
		}
		return dst;
	}
}
