//针对灰度图的中值滤波+CVPR 2019的SideWindowFilter
//其他种类的滤波直接换核即可

//记录每一个方向的核的不为0的元素个数
int cnt[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
// 记录每一个方向的滤波器
vector <int> filter[8];

//初始化半径为radius的滤波器，原理可以看https://mp.weixin.qq.com/s/vjzZjRoQw7MnkqAfvwBUNA
void InitFilter(int radius) {
	int n = radius * 2 + 1;
	for (int i = 0; i < 8; i++) {
		cnt[i] = 0;
		filter[i].clear();
	}
	for (int i = 0; i < 8; i++) {
		for (int x = 0; x < n; x++) {
			for (int y = 0; y < n; y++) {
				if (i == 0 && x <= radius && y <= radius) {
					filter[i].push_back(1);
				}
				else if (i == 1 && x <= radius && y >= radius) {
					filter[i].push_back(1);
				}
				else if (i == 2 && x >= radius && y <= radius) {
					filter[i].push_back(1);
				}
				else if (i == 3 && x >= radius && y >= radius) {
					filter[i].push_back(1);
				}
				else if (i == 4 && x <= radius) {
					filter[i].push_back(1);
				}
				else if (i == 5 && x >= radius) {
					filter[i].push_back(1);
				}
				else if (i == 6 && y >= radius) {
					filter[i].push_back(1);
				}
				else if (i == 7 && y <= radius) {
					filter[i].push_back(1);
				}
				else {
					filter[i].push_back(0);
				}
			}
		}
	}
	for (int i = 0; i < 8; i++) {
		int sum = 0;
		for (int j = 0; j < filter[i].size(); j++) sum += filter[i][j] == 1;
		cnt[i] = sum;
	}
}

//实现Side Window Filter的中值滤波，强制保边
Mat MedianSideWindowFilter(Mat src, int radius = 1) {
	int row = src.rows;
	int col = src.cols;
	int channels = src.channels();
	InitFilter(radius);
	//针对灰度图
	if (channels == 1) {
		Mat dst(row, col, CV_8UC1);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (i < radius || i + radius >= row || j < radius || j + radius >= col) {
					dst.at<uchar>(i, j) = src.at<uchar>(i, j);
					continue;
				}
				int minn = 256;
				int pos = 0;
				for (int k = 0; k < 8; k++) {
					int val = 0;
					int id = 0;
					vector <int> now;
					for (int x = -radius; x <= radius; x++) {
						for (int y = -radius; y <= radius; y++) {
							//if (x == 0 && y == 0) continue;
							now.push_back(src.at<uchar>(i + x, j + y) * filter[k][id++]);
							//val += src.at<uchar>(i + x, j + y) * filter[k][id++];
						}
					}
					sort(now.begin(), now.end());
					val = now[(2 * radius + 1)*(2 * radius + 1) / 2];
					if (abs(val - src.at<uchar>(i, j)) < minn) {
						minn = abs(val - src.at<uchar>(i, j));
						pos = k;
					}
				}
				int val = 0;
				int id = 0;
				vector <int> now;
				for (int x = -radius; x <= radius; x++) {
					for (int y = -radius; y <= radius; y++) {
						//if (x == 0 && y == 0) continue;
						now.push_back(src.at<uchar>(i + x, j + y) * filter[pos][id++]);
						//val += src.at<uchar>(i + x, j + y) * filter[k][id++];
					}
				}
				sort(now.begin(), now.end());
				val = now[(2 * radius + 1)*(2 * radius + 1) / 2];
				dst.at<uchar>(i, j) = val;
			}
		}
		return dst;
	}
	//针对RGB图
	Mat dst(row, col, CV_8UC3);
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (i < radius || i + radius >= row || j < radius || j + radius >= col) {
					dst.at<Vec3b>(i, j)[c] = src.at<Vec3b>(i, j)[c];
					continue;
				}
				int minn = 256;
				int pos = 0;
				for (int k = 0; k < 8; k++) {
					int val = 0;
					int id = 0;
					vector <int> now;
					for (int x = -radius; x <= radius; x++) {
						for (int y = -radius; y <= radius; y++) {
							//if (x == 0 && y == 0) continue;
							//val += src.at<Vec3b>(i + x, j + y)[c] * filter[k][id++];
							now.push_back(src.at<Vec3b>(i + x, j + y)[c] * filter[k][id++]);
						}
					}
					val = now[(2 * radius + 1)*(2 * radius + 1) / 2];
					if (abs(val - src.at<Vec3b>(i, j)[c]) < minn) {
						minn = abs(val - src.at<Vec3b>(i, j)[c]);
						pos = k;
					}
				}
				int val = 0;
				int id = 0;
				vector <int> now;
				for (int x = -radius; x <= radius; x++) {
					for (int y = -radius; y <= radius; y++) {
						//if (x == 0 && y == 0) continue;
						//val += src.at<Vec3b>(i + x, j + y)[c] * filter[k][id++];
						now.push_back(src.at<Vec3b>(i + x, j + y)[c] * filter[pos][id++]);
					}
				}
				val = now[(2 * radius + 1)*(2 * radius + 1) / 2];
				dst.at<Vec3b>(i, j)[c] = val;
			}
		}
	}
	return dst;
}


int main() {
	Mat src = imread("F:\\1.jpg");
	for (int i = 0; i < 10; i++) {
		src = MedianSideWindowFilter(src, 3);
	}
	imwrite("F:\\res.jpg", src);
	return 0;
}