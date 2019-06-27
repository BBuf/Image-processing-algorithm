//8方向浮雕
enum  DIR
{
	N,
	NE,
	E,
	SE,
	S,
	SW,
	W,
	NW
};

Mat DiamondEmboss(Mat src, DIR dir = SE, int offset = 127) {
	int row = src.rows;
	int col = src.cols;
	int ioffset = 0;
	int joffset = 0;
	switch (dir)
	{
	case N:
		ioffset = -1;
		joffset = 0;
		break;
	case NE:
		ioffset = -1;
		joffset = 1;
		break;
	case E:
		ioffset = 0;
		joffset = 1;
		break;
	case SE:
		ioffset = 1;
		joffset = 1;
		break;
	case S:
		ioffset = 1;
		joffset = -1;
		break;
	case SW:
		ioffset = 0;
		joffset = -1;
		break;
	case W:
		ioffset = 0;
		joffset = -1;
		break;
	case NW:
		ioffset = 1;
		joffset = 1;
		break;
	default:
		break;
	}
	Mat dst(row, col, CV_8UC3);
	int border = 1;
	for (int i = border; i < row - border; i++) {
		for (int j = border; j < col - border; j++) {
			for (int k = 0; k < 3; k++) {
				int sum = src.at<Vec3b>(i, j)[k] - src.at<Vec3b>(i - ioffset, j - joffset)[k] + offset;
				if (sum < 0) sum = -sum;
				if (sum < 64) sum = 64;
				if (sum > 255) sum = 255;
				dst.at<Vec3b>(i, j)[k] = sum;
			}
		}
	}
	return dst;
}