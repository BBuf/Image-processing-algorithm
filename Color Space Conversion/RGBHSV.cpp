Mat RGB2HSV(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_32FC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float b = src.at<Vec3b>(i, j)[0] / 255.0;
			float g = src.at<Vec3b>(i, j)[1] / 255.0;
			float r = src.at<Vec3b>(i, j)[2] / 255.0;
			float minn = min(r, min(g, b));
			float maxx = max(r, max(g, b));
			dst.at<Vec3f>(i, j)[2] = maxx; //V
			float delta = maxx - minn;
			float h, s;
			if (maxx != 0) {
				s = delta / maxx;
			}
			else {
				s = 0;
			}
			if (r == maxx) {
				h = (g - b) / delta;
			}
			else if (g == maxx) {
				h = 2 + (b - r) / delta;
			}
			else {
				h = 4 + (r - g) / delta;
			}
			h *= 60;
			if (h < 0)
				h += 360;
			dst.at<Vec3f>(i, j)[0] = h;
			dst.at<Vec3f>(i, j)[1] = s;
		}
	}
	return dst;
}

Mat HSV2RGB(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	float r, g, b, h, s, v;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			h = src.at<Vec3f>(i, j)[0];
			s = src.at<Vec3f>(i, j)[1];
			v = src.at<Vec3f>(i, j)[2];
			if (s == 0) {
				r = g = b = v;
			}
			else {
				h /= 60;
				int offset = floor(h);
				float f = h - offset;
				float p = v * (1 - s);
				float q = v * (1 - s * f);
				float t = v * (1 - s * (1 - f));
				switch (offset)
				{
				case 0: r = v; g = t; b = p; break;
				case 1: r = q; g = v; b = p; break;
				case 2: r = p; g = v; b = t; break;
				case 3: r = p; g = q; b = v; break;
				case 4: r = t; g = p; b = v; break;
				case 5: r = v; g = p; b = q; break;
				default:
					break;
				}
			}
			dst.at<Vec3b>(i, j)[0] = int(b * 255);
			dst.at<Vec3b>(i, j)[1] = int(g * 255);
			dst.at<Vec3b>(i, j)[2] = int(r * 255);
		}
	}
	return dst;
}