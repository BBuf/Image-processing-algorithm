Mat BicubicInterpolation(Mat src, float scale_x, float scale_y) {
	int row = src.rows;
	int col = src.cols;
	int channels = src.channels();
	int dst_row = round(row * scale_x);
	int dst_col = round(col * scale_y);
	Mat dst(dst_row, dst_col, CV_8UC3, Scalar::all(0));
#pragma omp parallel for num_threads(4)
	for (int j = 0; j < dst_row; j++) {
		float fy = (float)((j + 0.5) / scale_y - 0.5);
		int sy = cvFloor(fy);
		fy -= sy;
		sy = min(sy, row - 3);
		sy = max(1, sy);

		const float A = -0.75f;

		float coeffsY[4];
		coeffsY[0] = ((A*(fy + 1) - 5 * A)*(fy + 1) + 8 * A)*(fy + 1) - 4 * A;
		coeffsY[1] = ((A + 2)*fy - (A + 3))*fy*fy + 1;
		coeffsY[2] = ((A + 2)*(1 - fy) - (A + 3))*(1 - fy)*(1 - fy) + 1;
		coeffsY[3] = 1.f - coeffsY[0] - coeffsY[1] - coeffsY[2];

		short cbufY[4];
		cbufY[0] = cv::saturate_cast<short>(coeffsY[0] * 2048);
		cbufY[1] = cv::saturate_cast<short>(coeffsY[1] * 2048);
		cbufY[2] = cv::saturate_cast<short>(coeffsY[2] * 2048);
		cbufY[3] = cv::saturate_cast<short>(coeffsY[3] * 2048);
		for (int i = 0; i < dst_col; i++) {
			float fx = (float)((i + 0.5) / scale_x - 0.5);
			int sx = cvFloor(fx);
			fx -= sx;

			if (sx < 1) {
				fx = 0, sx = 1;
			}
			if (sx >= col - 3) {
				fx = 0, sx = col - 3;
			}

			float coeffsX[4];
			coeffsX[0] = ((A*(fx + 1) - 5 * A)*(fx + 1) + 8 * A)*(fx + 1) - 4 * A;
			coeffsX[1] = ((A + 2)*fx - (A + 3))*fx*fx + 1;
			coeffsX[2] = ((A + 2)*(1 - fx) - (A + 3))*(1 - fx)*(1 - fx) + 1;
			coeffsX[3] = 1.f - coeffsX[0] - coeffsX[1] - coeffsX[2];

			short cbufX[4];
			cbufX[0] = cv::saturate_cast<short>(coeffsX[0] * 2048);
			cbufX[1] = cv::saturate_cast<short>(coeffsX[1] * 2048);
			cbufX[2] = cv::saturate_cast<short>(coeffsX[2] * 2048);
			cbufX[3] = cv::saturate_cast<short>(coeffsX[3] * 2048);

			for (int k = 0; k < channels; ++k)
			{
				dst.at<Vec3b>(j, i)[k] = abs((src.at<Vec3b>(sy - 1, sx - 1)[k] * cbufX[0] * cbufY[0] + src.at<Vec3b>(sy, sx - 1)[k] * cbufX[0] * cbufY[1] +
					src.at<Vec3b>(sy + 1, sx - 1)[k] * cbufX[0] * cbufY[2] + src.at<Vec3b>(sy + 2, sx - 1)[k] * cbufX[0] * cbufY[3] +
					src.at<Vec3b>(sy - 1, sx)[k] * cbufX[1] * cbufY[0] + src.at<Vec3b>(sy, sx)[k] * cbufX[1] * cbufY[1] +
					src.at<Vec3b>(sy + 1, sx)[k] * cbufX[1] * cbufY[2] + src.at<Vec3b>(sy + 2, sx)[k] * cbufX[1] * cbufY[3] +
					src.at<Vec3b>(sy - 1, sx + 1)[k] * cbufX[2] * cbufY[0] + src.at<Vec3b>(sy, sx + 1)[k] * cbufX[2] * cbufY[1] +
					src.at<Vec3b>(sy + 1, sx + 1)[k] * cbufX[2] * cbufY[2] + src.at<Vec3b>(sy + 2, sx + 1)[k] * cbufX[2] * cbufY[3] +
					src.at<Vec3b>(sy - 1, sx + 2)[k] * cbufX[3] * cbufY[0] + src.at<Vec3b>(sy, sx + 2)[k] * cbufX[3] * cbufY[1] +
					src.at<Vec3b>(sy + 1, sx + 2)[k] * cbufX[3] * cbufY[2] + src.at<Vec3b>(sy + 2, sx + 2)[k] * cbufX[3] * cbufY[3]) >> 22);
			}
		}
	}
	return dst;
}