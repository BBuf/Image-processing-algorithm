Mat MarrEdgeDetection(Mat src, int kernelDiameter, double sigma) {
	int kernel_size = kernelDiameter / 2;
	Mat kernel(kernelDiameter, kernelDiameter, CV_64FC1);
	for (int i = -kernel_size; i <= kernel_size; i++) {
		for (int j = -kernel_size; j <= kernel_size; j++) {
			kernel.at<double>(i + kernel_size, j + kernel_size) = exp(-((pow(j, 2) + pow(i, 2)) /
				(pow(sigma, 2) * 2)))
				* (((pow(j, 2) + pow(i, 2) - 2 *
					pow(sigma, 2)) / (2 * pow(sigma, 4))));
		}
	}
	Mat laplacian(src.rows - kernel_size * 2, src.cols - kernel_size * 2, CV_64FC1);
	Mat dst = Mat::zeros(src.rows - kernel_size * 2, src.cols - kernel_size * 2, CV_8UC1);
	for (int i = kernel_size; i < src.rows - kernel_size; i++) {
		for (int j = kernel_size; j < src.cols - kernel_size; j++) {
			double sum = 0;
			for (int x = -kernel_size; x <= kernel_size; x++){
				for (int y = -kernel_size; y <= kernel_size; y++) {
					sum +=  src.at<uchar>(i + x, j + y) * kernel.at<double>(x + kernel_size, y + kernel_size);
				}
			}
			laplacian.at<double>(i - kernel_size, j - kernel_size) = sum;
		}
	}
	for (int i = 1; i < dst.rows - 1; i++) {
		for (int j = 1; j < dst.cols - 1; j++) {
			if ((laplacian.at<double>(i - 1, j) * laplacian.at<double>(i + 1, j) < 0) || (laplacian.at<double>(i, j + 1) * laplacian.at<double>(i, j - 1) < 0) ||
				(laplacian.at<double>(i + 1, j - 1) * laplacian.at<double>(i - 1, j + 1) < 0) || (laplacian.at<double>(i - 1, j - 1) * laplacian.at <double> (i + 1, j + 1) < 0)) {
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
	return dst;
}