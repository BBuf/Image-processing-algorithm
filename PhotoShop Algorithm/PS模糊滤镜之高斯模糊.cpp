// 创建高斯核
// kSize:卷积核的大小3、5、7等（3×3、5×5、7×7）
// sigma:方差
const float EPS = 1e-7;
void CreatGaussKernel(float **pdKernel, int kSize, float sigma) {
	int sum = 0;
	float dCenter = (kSize - 1) / 2;
	//生成高斯数据
	for (int i = 0; i < kSize; i++) {
		for (int j = 0; j < kSize; j++) {
			//用和来近似平方和的开方
			float dis = fabsf(i - dCenter) + fabsf(j - dCenter);
			float val = exp(-dis * dis / (2 * sigma * sigma + EPS));
			pdKernel[i][j] = val;
			sum += val;
		}
	}
	//归一化
	for (int i = 0; i < kSize; i++) {
		for (int j = 0; j < kSize; j++) {
			pdKernel[i][j] /= (sum + EPS);
		}
	}
}

Mat GaussBlur(Mat src, int kSize, float sigma) {
	int row = src.rows;
	int col = src.cols;
	//分配高斯核空间
	float **pKernel = new float*[kSize];
	for (int i = 0; i < kSize; i++) {
		pKernel[i] = new float[kSize];
	}
	Mat dst(row, col, CV_8UC3);
	CreatGaussKernel(pKernel, kSize, sigma);
	int border = (kSize - 1) / 2;
	float sum = 0;
	for (int i = border; i < row - border; i++) {
		for (int j = border; j < col - border; j++) {
			for (int k = 0; k < 3; k++) {
				sum = 0;
				for (int x = -border; x <= border; x++) {
					for (int y = -border; y <= border; y++) {
						sum += src.at<Vec3b>(i + x, j + y)[k] * pKernel[border + x][border + y];
					}
				}
				if (sum > 255) sum = 255;
				else if (sum < 0) sum = 0;
				dst.at<Vec3b>(i, j)[k] = sum;
			}
		}
	}
	return dst;
}