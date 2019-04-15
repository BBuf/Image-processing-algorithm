Mat speed_BillnearInterpolation(Mat src, float sx, float sy) {
	int row = src.rows;
	int col = src.cols;
	int channels = src.channels();
	int dst_row = round(row * sx);
	int dst_col = round(col * sy);
	Mat dst(dst_row, dst_col, CV_8UC3);
#pragma parallel for num_threads(4)
	for (int i = 0; i < dst_row; i++) {
		int index_i = (i + 0.5) / sx - 0.5;
		if (index_i < 0) index_i = 0;
		if (index_i > row - 1) index_i = row - 1;
		int i1 = floor(index_i);
		int i2 = ceil(index_i);
		float u = index_i - i;
		short cy[2];
		cy[0] = saturate_cast<short>((1.f - u) * 2048);
		cy[1] = 2048 - cy[0];
		for (int j = 0; j < dst_col; j++) {
			float index_j = (j + 0.5) / sy - 0.5;
			if (index_j < 0) index_j = 0;
			if (index_j > col - 1) index_j = col - 1;
			int j1 = floor(index_j);
			int j2 = ceil(index_j);
			float v = index_j - j1;
			short cx[2];
			cx[0] = saturate_cast<short>((1.f - v) * 2048);
			cx[1] = 2048 - cx[0];
			for (int k = 0; k < 3; k++) {
				dst.at<cv::Vec3b>(i, j)[k] = (cy[0] * cx[0] * src.at<cv::Vec3b>(i1, j1)[k] +
					cy[0] * cx[1] * src.at<cv::Vec3b>(i1, j2)[k] + cy[1] * cx[0] * src.at<cv::Vec3b>(i2, j1)[k] + cy[1] * cx[1] * src.at<cv::Vec3b>(i2, j2)[k]) >> 22;
			}
		}
	}
	return dst;
}