Mat speed_rgb2gray(Mat src) {
	Mat dst(src.rows, src.cols, CV_8UC1);
#pragma omp parallel for num_threads(12)
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<uchar>(i, j) = ((src.at<Vec3b>(i, j)[0] << 18) + (src.at<Vec3b>(i, j)[0] << 15) + (src.at<Vec3b>(i, j)[0] << 14) +
				(src.at<Vec3b>(i, j)[0] << 11) + (src.at<Vec3b>(i, j)[0] << 7) + (src.at<Vec3b>(i, j)[0] << 7) + (src.at<Vec3b>(i, j)[0] << 5) +
				(src.at<Vec3b>(i, j)[0] << 4) + (src.at<Vec3b>(i, j)[0] << 2) +
				(src.at<Vec3b>(i, j)[1] << 19) + (src.at<Vec3b>(i, j)[1] << 16) + (src.at<Vec3b>(i, j)[1] << 14) + (src.at<Vec3b>(i, j)[1] << 13) +
				(src.at<Vec3b>(i, j)[1] << 10) + (src.at<Vec3b>(i, j)[1] << 8) + (src.at<Vec3b>(i, j)[1] << 4) + (src.at<Vec3b>(i, j)[1] << 3) + (src.at<Vec3b>(i, j)[1] << 1) +
				(src.at<Vec3b>(i, j)[2] << 16) + (src.at<Vec3b>(i, j)[2] << 15) + (src.at<Vec3b>(i, j)[2] << 14) + (src.at<Vec3b>(i, j)[2] << 12) +
				(src.at<Vec3b>(i, j)[2] << 9) + (src.at<Vec3b>(i, j)[2] << 7) + (src.at<Vec3b>(i, j)[2] << 6) + (src.at<Vec3b>(i, j)[2] << 5) + (src.at<Vec3b>(i, j)[2] << 4) + (src.at<Vec3b>(i, j)[2] << 1) >> 20);
		}
	}
	return dst;
}


Mat unevenLightCompensate(Mat src, int block_Size) {
	int row = src.rows;
	int col = src.cols;
	Mat gray(row, col, CV_8UC1);
	if (src.channels() == 3) {
		gray = speed_rgb2gray(src);
	}
	else {
		gray = src;
	}
	float average = mean(gray)[0];
	int new_row = ceil(1.0 * row / block_Size);
	int new_col = ceil(1.0 * col / block_Size);
	Mat new_img(new_row, new_col, CV_32FC1);
	for (int i = 0; i < new_row; i++) {
		for (int j = 0; j < new_col; j++) {
			int rowx = i * block_Size;
			int rowy = (i + 1) * block_Size;
			int colx = j * block_Size;
			int coly = (j + 1) * block_Size;
			if (rowy > row) rowy = row;
			if (coly > col) coly = col;
			Mat ROI = src(Range(rowx, rowy), Range(colx, coly));
			float block_average = mean(ROI)[0];
			new_img.at<float>(i, j) = block_average;
		}
	}
	new_img = new_img - average;
	Mat new_img2;
	resize(new_img, new_img2, Size(row, col), (0, 0), (0, 0), INTER_CUBIC);
	Mat new_src;
	gray.convertTo(new_src, CV_32FC1);
	Mat dst = new_src - new_img2;
	dst.convertTo(dst, CV_8UC1);
	return dst;
}