Mat speed_rgb2gray(Mat src) {
	Mat dst(src.rows, src.cols, CV_8UC1);
#pragma omp parallel for num_threads(4)
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

//积分图常规方法，求2维前缀和
Mat Normal(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst = Mat::zeros(row+1, col+1, CV_64F);
	for (int i = 1; i < dst.rows; i++) {
		for (int j = 1; j < dst.cols; j++) {
			double up_left = dst.at<double>(i - 1, j - 1);
			double up_right = dst.at<double>(i - 1, j);
			double bot_left = dst.at<double>(i, j - 1);
			int bot_right = src.at<uchar>(i - 1, j - 1);
			dst.at<double>(i, j) = (bot_right + bot_left + up_right - up_left);
		}
	}
	return dst;
}


//积分图优化方法，由上方src(i-1,j)加上当前行的和即可
Mat Fast(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst = Mat::zeros(row + 1, col + 1, CV_64F);
	int sum = 0;
	for (int i = 1; i < row + 1; i++) {
		sum = 0;
		for (int j = 1; j < col + 1; j++) {
			sum += src.at<uchar>(i - 1, j - 1);
			dst.at<double>(i, j) = (dst.at<double>(i - 1, j) + 1.0 * sum);
		}
	}
	return dst;
}

//均值滤波
Mat speed_MeanFilter(Mat src, int ksize) {
	Mat src_bordered;
	int row = src.rows;
	int col = src.cols;
	int radius = (ksize - 1) / 2;
	copyMakeBorder(src, src_bordered, radius, radius, radius, radius, BORDER_REFLECT101);
	Mat dst = Mat::zeros(row, col, CV_8UC1);
	Mat Table = Normal(src_bordered);
	for (int i = radius + 1; i < src.rows + radius + 1; i++) {
		for (int j = radius + 1; j < src.cols + radius + 1; j++) {
			double up_left = Table.at<double>(i - radius - 1, j - radius - 1);
			double up_right = Table.at<double>(i - radius - 1, j + radius);
			double bot_left = Table.at<double>(i + radius, j - radius - 1);
			double bot_right = Table.at<double>(i + radius, j + radius);
			double sum = (bot_right - up_right - bot_left + up_left);
			double mean = sum / (ksize * ksize);
			if (mean < 0) {
				mean = 0;
			}
			else if (mean > 255) {
				mean = 255;
			}
			dst.at<uchar>(i - radius - 1, j - radius - 1) = (int)mean;
		}
	}
	return dst;
}

Mat SerialMedianFiltering(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC1);
	for (int i = 1; i < row - 1; i++) {
		for (int j = 1; j < col - 1; j++) {
			dst.at<uchar>(i, j) = ((int)src.at<uchar>(i, j) + src.at<uchar>(i, j + 1) + src.at<uchar>(i, j - 1) +
				src.at<uchar>(i - 1, j) + src.at<uchar>(i - 1, j - 1) + src.at<uchar>(i - 1, j + 1) +
				src.at<uchar>(i + 1, j) + src.at<uchar>(i + 1, j - 1) + src.at<uchar>(i + 1, j + 1)) / 9;
		}
	}
	for (int i = 0; i < row; i++) {
		dst.at<uchar>(i, 0) = src.at<uchar>(i, 0);
		dst.at<uchar>(i, col - 1) = src.at<uchar>(i, col - 1);
	}
	for (int i = 0; i < col; i++) {
		dst.at<uchar>(0, i) = src.at<uchar>(0, i);
		dst.at<uchar>(row - 1, i) = src.at<uchar>(row - 1, i);
	}
	return dst;
}
