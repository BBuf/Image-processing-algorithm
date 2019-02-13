Mat RGB2HSI(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_32FC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float b = src.at<Vec3b>(i, j)[0] / 255.0;
			float g = src.at<Vec3b>(i, j)[1] / 255.0;
			float r = src.at<Vec3b>(i, j)[2] / 255.0;
			float minn = min(b, min(g, r));
			float maxx = max(b, max(g, r));
			float H = 0;
			float S = 0;
			float I = (minn + maxx) / 2.0f;
			if (maxx == minn) {
				dst.at<Vec3f>(i, j)[0] = H;
				dst.at<Vec3f>(i, j)[1] = S;
				dst.at<Vec3f>(i, j)[2] = I;
			}
			else {
				float delta = maxx - minn;
				if (I < 0.5) {
					S = delta / (maxx + minn);
				}
				else {
					S = delta / (2.0 - maxx - minn);
				}
				if (r == maxx) {
					if (g > b) {
						H = (g - b) / delta;
					}
					else {
						H = 6.0 + (g - b) / delta;
					}
				}
				else if (g == maxx) {
					H = 2.0 + (b - r) / delta;
				}
				else {
					H = 4.0 + (r - g) / delta;
				}
				H /= 6.0; //除以6，表示在那个部分
				if (H < 0.0)
					H += 1.0;
				if (H > 1)
					H -= 1;
				H = (int)(H * 360); //转成[0, 360]
				dst.at<Vec3f>(i, j)[0] = H;
				dst.at<Vec3f>(i, j)[1] = S;
				dst.at<Vec3f>(i, j)[2] = I;
			}
		}
	}
	return dst;
}

float get_Ans(double p, double q, double Ht) {
	if (Ht < 0.0)
		Ht += 1.0;
	else if (Ht > 1.0)
		Ht -= 1.0;
	if ((6.0 * Ht) < 1.0)
		return (p + (q - p) * Ht * 6.0);
	else if ((2.0 * Ht) < 1.0)
		return q;
	else if ((3.0 * Ht) < 2.0)
		return (p + (q - p) * ((2.0F / 3.0F) - Ht) * 6.0);
	else
		return (p);
}

Mat HSI2RGB(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float r, g, b, M1, M2;
			float H = src.at<Vec3f>(i, j)[0];
			float S = src.at<Vec3f>(i, j)[1];
			float I = src.at<Vec3f>(i, j)[2];
			float hue = H / 360;
			if (S == 0) {//灰色
				r = g = b = I;
			}
			else {
				if (I <= 0.5) {
					M2 = I * (1.0 + S);
				}
				else {
					M2 = I + S - I * S;
				}
				M1 = (2.0 * I - M2);
				r = get_Ans(M1, M2, hue + 1.0 / 3.0);
				g = get_Ans(M1, M2, hue);
				b = get_Ans(M1, M2, hue - 1.0 / 3.0);
			}
			dst.at<Vec3b>(i, j)[0] = (int)(b * 255);
			dst.at<Vec3b>(i, j)[1] = (int)(g * 255);
			dst.at<Vec3b>(i, j)[2] = (int)(r * 255);
		}
	}
	return dst;
}
