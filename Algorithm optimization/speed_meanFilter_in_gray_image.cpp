//所有代码针对灰度图，RGB分为3个通道处理
//中值滤波串行代码
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

//x86循环展开优化均值滤波
Mat x86MedianFiltering(Mat src, const int BX, const int BY) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC1);
	int i, j;
	for (i = 1; i < row - 1 - BY; i += BY) {
		for (j = 1; j < col - 1 - BX; j += BX) {
			unsigned short *temp = new unsigned short[BX*BY];
			memset(temp, 0, sizeof(temp));
			for (int ii = 0; ii < BY; ii++) {
				for (int jj = 0; jj < BX; jj++) {
					temp[ii*BX + jj] += src.at<uchar>(i + ii, j + jj - 1);
					temp[ii*BX + jj] += src.at<uchar>(i + ii, j + jj);
					temp[ii*BX + jj] += src.at<uchar>(i + ii, j + jj + 1);
					temp[ii*BX + jj] += src.at<uchar>(i + ii + 1, j + jj - 1);
					temp[ii*BX + jj] += src.at<uchar>(i + ii + 1, j + jj);
					temp[ii*BX + jj] += src.at<uchar>(i + ii + 1, j + jj + 1);
					temp[ii*BX + jj] += src.at<uchar>(i + ii - 1, j + jj - 1);
					temp[ii*BX + jj] += src.at<uchar>(i + ii - 1, j + jj);
					temp[ii*BX + jj] += src.at<uchar>(i + ii - 1, j + jj + 1);
				}
			}
			for (int ii = 0; ii < BY; ii++) {
				for (int jj = 0; jj < BX; jj++) {
					dst.at<uchar>(i + ii, j + jj) = (unsigned char)(temp[ii * BX + jj] / 9);
				}
			}
		}
		for (int ii = 0; ii < BY; ii++) {
			for (int je = j; je < col - 1; je++) {
				unsigned short temp = 0;
				temp += src.at<uchar>(i + ii, je);
				temp += src.at<uchar>(i + ii, je + 1);
				temp += src.at<uchar>(i + ii, je - 1);

				temp += src.at<uchar>(i + ii + 1, je);
				temp += src.at<uchar>(i + ii + 1, je + 1);
				temp += src.at<uchar>(i + ii + 1, je - 1);

				temp += src.at<uchar>(i + ii - 1, je);
				temp += src.at<uchar>(i + ii - 1, je + 1);
				temp += src.at<uchar>(i + ii - 1, je - 1);
				dst.at<uchar>(i + ii, je) = (unsigned char)(temp / 9);
			}
		}
	}
	for (int ie = i; ie < row - 1; ie++) {
		for (int j = 1; j < col - 1; j++) {
			unsigned short temp = 0;
			temp += src.at<uchar>(ie, j);
			temp += src.at<uchar>(ie, j + 1);
			temp += src.at<uchar>(ie, j - 1);
			temp += src.at<uchar>(ie - 1, j);
			temp += src.at<uchar>(ie - 1, j + 1);
			temp += src.at<uchar>(ie - 1, j - 1);
			temp += src.at<uchar>(ie + 1, j);
			temp += src.at<uchar>(ie + 1, j + 1);
			temp += src.at<uchar>(ie + 1, j - 1);
			dst.at<uchar>(ie, j) = (unsigned char)(temp / 9);
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
