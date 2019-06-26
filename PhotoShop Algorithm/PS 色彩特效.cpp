Mat ColorTrans(Mat src, int op) {
	//op=1 呈现碧绿效果，就是要使色彩呈暗绿色，给人诡异的感觉
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	if (op == 1) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				int b = src.at<Vec3b>(i, j)[0];
				int g = src.at<Vec3b>(i, j)[1];
				int r = src.at<Vec3b>(i, j)[2];
				int R = (g - b) * (g - b) / 128;
				int G = (r - b) * (r - b) / 128;
				int B = (r - g) * (r - g) / 128;
				if (R > 255) R = 255;
				if (R < 0) R = 0;
				if (G > 255) G = 255;
				if (G < 0) G = 0;
				if (B > 255) B = 255;
				if (B < 0) B = 0;
				dst.at<Vec3b>(i, j)[0] = B;
				dst.at<Vec3b>(i, j)[1] = G;
				dst.at<Vec3b>(i, j)[2] = R;
			}
		}
	}
	else if (op == 2) {
		//op=2 棕褐色效果就是要实现那种图像模糊、略带发黄的老照片的感觉，别具风情
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				int b = src.at<Vec3b>(i, j)[0];
				int g = src.at<Vec3b>(i, j)[1];
				int r = src.at<Vec3b>(i, j)[2];
				int R = 0.393 * r + 0.769 * g + 0.189 * b;
				int G = 0.349 * r + 0.686 * g + 0.168 * b;
				int B = 0.272 * r + 0.534 * g + 0.131 * b;
				if (R > 255) R = 255;
				if (R < 0) R = 0;
				if (G > 255) G = 255;
				if (G < 0) G = 0;
				if (B > 255) B = 255;
				if (B < 0) B = 0;
				dst.at<Vec3b>(i, j)[0] = B;
				dst.at<Vec3b>(i, j)[1] = G;
				dst.at<Vec3b>(i, j)[2] = R;
			}
		}
	}
	else if (op == 3) {
		//op=3 冰冻效果就是使图像呈现一种晶莹的淡蓝色
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				int b = src.at<Vec3b>(i, j)[0];
				int g = src.at<Vec3b>(i, j)[1];
				int r = src.at<Vec3b>(i, j)[2];
				int R = abs(r - g - b) * 3 / 2;
				int G = abs(g - b - r) * 3 / 2;
				int B = abs(b - r - g) * 3 / 2;
				if (R > 255) R = 255;
				if (R < 0) R = 0;
				if (G > 255) G = 255;
				if (G < 0) G = 0;
				if (B > 255) B = 255;
				if (B < 0) B = 0;
				dst.at<Vec3b>(i, j)[0] = B;
				dst.at<Vec3b>(i, j)[1] = G;
				dst.at<Vec3b>(i, j)[2] = R;
			}
		}
	}
	else if (op == 4) {
		//op=4 熔铸效果类似打铁的场景
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				int b = src.at<Vec3b>(i, j)[0];
				int g = src.at<Vec3b>(i, j)[1];
				int r = src.at<Vec3b>(i, j)[2];
				int R = r * 128 / (g + b + 1);
				int G = g * 128 / (b + r + 1);
				int B = b * 128 / (r + g + 1);
				if (R > 255) R = 255;
				if (R < 0) R = 0;
				if (G > 255) G = 255;
				if (G < 0) G = 0;
				if (B > 255) B = 255;
				if (B < 0) B = 0;
				dst.at<Vec3b>(i, j)[0] = B;
				dst.at<Vec3b>(i, j)[1] = G;
				dst.at<Vec3b>(i, j)[2] = R;
			}
		}
	}
	else if (op == 5) {
		//op=5 暗调效果是通过降低色彩的各个分量，使整幅图像变得深谙
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				int b = src.at<Vec3b>(i, j)[0];
				int g = src.at<Vec3b>(i, j)[1];
				int r = src.at<Vec3b>(i, j)[2];
				int R = r * r / 255;
				int G = g * g / 255;
				int B = b * b / 255;
				/*if (R > 255) R = 255;
				if (R < 0) R = 0;
				if (G > 255) G = 255;
				if (G < 0) G = 0;
				if (B > 255) B = 255;
				if (B < 0) B = 0;*/
				dst.at<Vec3b>(i, j)[0] = B;
				dst.at<Vec3b>(i, j)[1] = G;
				dst.at<Vec3b>(i, j)[2] = R;
			}
		}
	}
	else if (op == 6) {
		//op=6 对调效果通过对彩色分量进行轮换的方式重新组合得到新的色彩
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				int b = src.at<Vec3b>(i, j)[0];
				int g = src.at<Vec3b>(i, j)[1];
				int r = src.at<Vec3b>(i, j)[2];
				int R = g * b / 255;
				int G = b * r / 255;
				int B = r * g / 255;
				/*if (R > 255) R = 255;
				if (R < 0) R = 0;
				if (G > 255) G = 255;
				if (G < 0) G = 0;
				if (B > 255) B = 255;
				if (B < 0) B = 0;*/
				dst.at<Vec3b>(i, j)[0] = B;
				dst.at<Vec3b>(i, j)[1] = G;
				dst.at<Vec3b>(i, j)[2] = R;
			}
		}
	}
	return dst;
}