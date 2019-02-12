//RGB2YUV优化
Mat speed_rgb2yuv(Mat src) {
	Mat dst(src.rows, src.cols, CV_8UC3);
#pragma omp parallel for num_threads(4)
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<Vec3b>(i, j)[0] =
				((src.at<Vec3b>(i, j)[2] << 6) + (src.at<Vec3b>(i, j)[2] << 3) + (src.at<Vec3b>(i, j)[2] << 2) + src.at<Vec3b>(i, j)[2] +
				(src.at<Vec3b>(i, j)[1] << 7) + (src.at<Vec3b>(i, j)[1] << 4) + (src.at<Vec3b>(i, j)[1] << 2) + (src.at<Vec3b>(i, j)[1] << 1) +
					(src.at<Vec3b>(i, j)[0] << 4) + (src.at<Vec3b>(i, j)[0] << 3) + (src.at<Vec3b>(i, j)[0] << 2) + src.at<Vec3b>(i, j)[0]) >> 8;
			dst.at<Vec3b>(i, j)[1] = (-((src.at<Vec3b>(i, j)[2] << 5) + (src.at<Vec3b>(i, j)[2] << 2) + (src.at<Vec3b>(i, j)[2] << 1)) -
				((src.at<Vec3b>(i, j)[1] << 6) + (src.at<Vec3b>(i, j)[1] << 3) + (src.at<Vec3b>(i, j)[1] << 1)) +
				((src.at<Vec3b>(i, j)[0] << 6) + (src.at<Vec3b>(i, j)[0] << 5) + (src.at<Vec3b>(i, j)[0] << 4))) >> 8;
			dst.at<Vec3b>(i, j)[2] = ((src.at<Vec3b>(i, j)[2] << 7) + (src.at<Vec3b>(i, j)[2] << 4) + (src.at<Vec3b>(i, j)[2] << 3) + (src.at<Vec3b>(i, j)[2] << 2) + (src.at<Vec3b>(i, j)[2] << 1) -
				((src.at<Vec3b>(i, j)[1] << 7) + (src.at<Vec3b>(i, j)[1] << 2)) - ((src.at<Vec3b>(i, j)[0] << 4) + (src.at<Vec3b>(i, j)[0] << 3) + (src.at<Vec3b>(i, j)[0] << 1))) >> 8;
		}
	}
	return dst;
}


//YUV2RGB优化
Mat speed_yuv2rgb(Mat src) {
	Mat dst(src.rows, src.cols, CV_8UC3);
	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<Vec3b>(i, j)[0] = ((src.at<Vec3b>(i, j)[0] << 8) + (src.at<Vec3b>(i, j)[1] << 9) + (src.at<Vec3b>(i, j)[1] << 3)) >> 8;
			dst.at<Vec3b>(i, j)[1] = ((src.at<Vec3b>(i, j)[0] << 8) - ((src.at<Vec3b>(i, j)[1] << 6) + (src.at<Vec3b>(i, j)[1] << 5) +
			(src.at<Vec3b>(i, j)[1] << 2)) - ((src.at<Vec3b>(i, j)[2] << 7) + (src.at<Vec3b>(i, j)[2] << 4) +
			(src.at<Vec3b>(i, j)[2] << 2) + src.at<Vec3b>(i, j)[2])) >> 8;
			dst.at<Vec3b>(i, j)[2] = ((src.at<Vec3b>(i, j)[0] << 8) + ((src.at<Vec3b>(i, j)[2] << 8) + (src.at<Vec3b>(i, j)[2] << 5) +
			(src.at<Vec3b>(i, j)[2] << 2))) >> 8;
		}
	}
	return dst;
}