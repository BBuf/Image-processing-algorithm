//针对灰度图的均值滤波+CVPR 2019的SideWindowFilter
//其他种类的滤波直接换核即可

Mat SideWindowFilter(Mat src, int radius = 1) {
	int row = src.rows;
	int col = src.cols;
	int cnt[8] = { 4, 4, 4, 4, 6, 6, 6, 6 };
	//3*3 的模板，如果半径不为1，那么需要修改
	int filter[8][9] = { { 1, 1, 0, 1, 1, 0, 0, 0, 0 },
	{ 0, 1, 1, 0, 1, 1, 0, 0, 0 },
	{ 0 ,0, 0, 1, 1, 0, 1, 1, 0 },
	{ 0, 0, 0, 0, 1, 1, 0, 1, 1 },
	{ 1, 1, 1, 1, 1, 1, 0, 0, 0 },
	{ 0, 0, 0, 1, 1, 1, 1, 1, 1 },
	{ 0, 1, 1, 0, 1, 1, 0, 1, 1 },
	{ 1, 1, 0, 1, 1, 0, 1, 1, 0 } };
	Mat dst(row, col, CV_8UC1);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (i < radius || i + radius > row || j < radius || j + radius > col) {
				dst.at<uchar>(i, j) = src.at<uchar>(i, j);
				continue;
			}
			int minn = 256;
			int pos = 0;
			for (int k = 0; k < 8; k++) {
				int val = 0;
				int id = 0;
				for (int x = -radius; x <= radius; x++) {
					for (int y = -radius; y <= radius; y++) {
						if (x == 0 && y == 0) continue;
						val += src.at<uchar>(i + x, j + y) * filter[k][id++];
					}
				}
				val /= cnt[k];
				if (abs(val - src.at<uchar>(i, j)) < minn) {
					minn = abs(val - src.at<uchar>(i, j));
					pos = k;
				}
			}
			int val = 0;
			int id = 0;
			for (int x = -radius; x <= radius; x++) {
				for (int y = -radius; y <= radius; y++) {
					if (x == 0 && y == 0) continue;
					val += src.at<uchar>(i + x, j + y) * filter[pos][id++];
				}
			}
			dst.at<uchar>(i, j) = val / cnt[pos];
		}
	}
	return dst;
}

int main() {
	Mat src = imread("F:\\cat.jpg", 0);
	for (int i = 0; i < 18; i++) {
		src = SideWindowFilter(src, 1);
		//medianBlur(src, src, 3);
	}
	//Mat dst;
	//medianBlur(src, dst, 3);
	Mat dst = SideWindowFilter(src, 1);
	imshow("result", dst);
	imwrite("F:\\res.jpg", dst);
	waitKey(0);
}