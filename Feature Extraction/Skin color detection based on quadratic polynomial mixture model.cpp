Mat SkinDetection(Mat src) {
	int row = src.rows;
	int col = src.cols;
	int channels = src.channels();
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			for (int k = 0; k < channels; k++) {
				dst.at<Vec3b>(i, j)[k] = src.at<Vec3b>(i, j)[k];
				//dst.at<Vec3b>(i, j)[k] = 0;
			}
			int B = src.at<Vec3b>(i, j)[0];
			int G = src.at<Vec3b>(i, j)[1];
			int R = src.at<Vec3b>(i, j)[2];
			if (R - G >= 45) {
				if (G > B) {
					int Sum = R + G + B;
					int T1 = 156 * R - 52 * Sum;
					int T2 = 156 * G - 52 * Sum;
					if (T1 * T1 + T2 * T2 >= (Sum * Sum) >> 4) {
						T1 = 10000 * G * Sum;
						int Lower = -7760 * R * R + 5601 * R * Sum + 1766 * Sum * Sum;
						if (T1 > Lower) {
							int Upper = -13767 * R * R + 10743 * R * Sum + 1452 * Sum * Sum;
							if (T1 < Upper) {
								for (int k = 0; k < channels; k++) {
									dst.at<Vec3b>(i, j)[k] = 255;
								}
							}
						}
					}
				}
			}
		}
	}
	return dst;
}