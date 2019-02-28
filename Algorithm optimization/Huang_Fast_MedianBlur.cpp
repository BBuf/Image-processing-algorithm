//计算中值
int getMediaValue(const int hist[], int thresh) {
	int sum = 0;
	for (int i = 0; i < 256; i++) {
		sum += hist[i];
		if (sum >= thresh) {
			return i;
		}
	}
	return 255;
}
//快速中值滤波，灰度图
Mat fastMedianBlur(Mat src, int diameter) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC1);
	int Hist[256] = { 0 };
	int radius = (diameter - 1) / 2;
	int windowSize = diameter * diameter;
	int threshold = windowSize / 2 + 1;
	uchar *srcData = src.data;
	uchar *dstData = dst.data;
	int right = col - radius;
	int bot = row - radius;
	for (int j = radius; j < bot; j++) {
		for (int i = radius; i < right; i++) {
			//每一行第一个待滤波元素建立直方图
			if (i == radius) {
				memset(Hist, 0, sizeof(Hist));
				for (int y = j - radius; y <= min(j + radius, row); y++) {
					for (int x = i - radius; x <= min(i + radius, col); x++) {
						uchar val = srcData[y * col + x];
						Hist[val]++;
					}
				}
			}
			else {
				int L = i - radius - 1;
				int R = i + radius;
				for (int y = j - radius; y <= min(j + radius, row); y++) {
					//更新左边一列
					Hist[srcData[y * col + L]]--;
					//更新右边一列
					Hist[srcData[y * col + R]]++;
				}
			}
			uchar medianVal = getMediaValue(Hist, threshold);
			dstData[j * col + i] = medianVal;
		}
	}
	//边界直接赋值
	for (int i = 0; i < col; i++) {
		for (int j = 0; j < radius; j++) {
			int id1 = j * col + i;
			int id2 = (row - j - 1) * col + i;
			dstData[id1] = srcData[id1];
			dstData[id2] = srcData[id2];
		}
	}
	for (int i = radius; i < row - radius - 1; i++) {
		for (int j = 0; j < radius; j++) {
			int id1 = i * col + j;
			int id2 = i * col + col - j - 1;
			dstData[id1] = srcData[id1];
			dstData[id2] = srcData[id2];
		}
	}
	return dst;
}