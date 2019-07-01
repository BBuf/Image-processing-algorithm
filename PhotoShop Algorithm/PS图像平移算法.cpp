// 移出的区域填充255
//tx：X方向的平移量
//ty: Y方向的平移量
//flag: 是否扩大图像，0：不扩大，1：扩大
Mat Translate(Mat src, int tx, int ty, bool flag) {
	int row = src.rows;
	int col = src.cols;
	if (flag) {//扩大
		Mat dst(row + abs(ty), col + abs(tx), CV_8UC3);
		int height = dst.rows;
		int width = dst.cols;
		for (int i = 0; i < height; i++) {
			int i0 = i - ty;
			if(i0 >= 0 && i0 < height){
				for (int j = 0; j < width; j++) {
					int j0 = j - tx;
					if (j0 >= 0 && j0 < col && i0 >= 0 && i0 < row) {
						for (int k = 0; k < 3; k++) {
							dst.at<Vec3b>(i, j)[k] = src.at<Vec3b>(i0, j0)[k];
						}
					}
					else {
						for (int k = 0; k < 3; k++) {
							dst.at<Vec3b>(i, j)[k] = 255;
						}
					}
				}
			}
			else {
				for (int j = 0; j < width; j++) {
					for (int k = 0; k < 3; k++) {
						dst.at<Vec3b>(i, j)[k] = 255;
					}
				}
			}
		}
		return dst;
	}
	else {
		Mat dst(row, col, CV_8UC3);
		if (tx < -col || tx > col || ty < -row || ty > row) { //整个移出图像区域，则直接返回
			dst = Scalar::all(0);
			return dst;
		}
		int height = dst.rows;
		int width = dst.cols;
		for (int i = 0; i < height; i++) {
			int i0 = i - ty;
			if (i0 >= 0 && i0 < height) {
				for (int j = 0; j < width; j++) {
					int j0 = j - tx;
					if (j0 >= 0 && j0 < width && i0 >= 0 && i0 < height) {
						for (int k = 0; k < 3; k++) {
							dst.at<Vec3b>(i, j)[k] = src.at<Vec3b>(i0, j0)[k];
						}
					}
					else {
						for (int k = 0; k < 3; k++) {
							dst.at<Vec3b>(i, j)[k] = 255;
						}
					}
				}
			}
			else {
				for (int j = 0; j < width; j++) {
					for (int k = 0; k < 3; k++) {
						dst.at<Vec3b>(i, j)[k] = 255;
					}
				}
			}
		}
		return dst;
	}
}
