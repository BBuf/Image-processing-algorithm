
//添加运动模糊效果
//angle:运动的方向,distance:运动的距离
//这里只是粗略的计算，以dx的长度为准，也可以以dy或者dx+dy等长度微赚
Mat MotionBlur(const Mat &src, int angle = 30, int distance = 100) {
	if (distance < 1) distance = 1;
	else if (distance > 200) distance = 200;
	double radian = ((double)angle + 180.0) / 180.0 * PI;
	int dx = (int)((double)distance * cos(radian) + 0.5);
	int dy = (int)((double)distance * sin(radian) + 0.5);
	int sign;
	if (dx < 0) sign = -1;
	if (dx > 0) sign = 1;
	int height = src.rows;
	int width = src.cols;
	int chns = src.channels();
	Mat dst;
	dst.create(height, width, src.type());
	for (int i = 0; i < height; i++) {
		unsigned  char* dstData = (unsigned char*)dst.data + dst.step * i;
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < chns; k++) {
				int sum = 0, count = 0;
				for (int p = 0; p < abs(dx); p++) {
					int i0 = i + p*sign;
					int j0 = j + p*sign;
					if (i0 >= 0 && i0 < height && j0 >= 0 && j0 < width) {
						count++;
						sum += src.at<Vec3b>(i0, j0)[k];
					}
				}
				if (count == 0) {
					dstData[j*chns + k] = src.at<Vec3b>(i, j)[k];
				}
				else {
					dstData[j*chns + k] = int(sum / (double)count + 0.5);
					if (dstData[j*chns + k] < 0) dstData[j*chns + k] = 0;
					else if (dstData[j*chns + k] > 255) dstData[j*chns + k] = 255;
				}
			}
		}
	}
	return dst;
}