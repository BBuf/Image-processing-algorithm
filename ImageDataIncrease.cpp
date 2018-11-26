//1. 旋转
Mat RotateImage(const Mat &img, int degree) {
	degree = -degree; //warpAffine默认的旋转方向是逆时针,所以加负号表示转化为顺时针
	double angle = degree * CV_PI / 180.; //弧度
	double a = sin(angle), b = cos(angle);
	int width = img.cols;
	int height = img.rows;
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));
	float map[6];
	Mat map_matrix = Mat(2, 3, CV_32F, map);
	//旋转中心
	CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);
	CvMat map_matrix2 = map_matrix;
	cv2DRotationMatrix(center, degree, 1.0, &map_matrix2);
	//Adjust rotation center to dst's center,
	// otherwise you will get only part of the result
	map[2] += (width_rotate - width) / 2;
	map[5] += (height_rotate - height) / 2;
	Mat img_rotate;
	//对图像做仿射变换
	warpAffine(img, img_rotate, map_matrix, Size(width_rotate, height_rotate), 1, 0, 0);
	return img_rotate;
}

//生成高斯分布随机数数列(Marsgglia和Bray在1964年提出)
double generateGaussianNoise(double mu, double sigma)
{
	static double V1, V2, S;
	static int phase = 0;
	double X;
	double U1, U2;
	if (phase == 0) {
		do {
			U1 = (double)rand() / RAND_MAX;
			U2 = (double)rand() / RAND_MAX;
			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while (S >= 1 || S == 0);
		X = V1 * sqrt(-2 * log(S) / S);
	}
	else {
		X = V2 * sqrt(-2 * log(S) / S);
	}
	phase = 1 - phase;
	return mu + sigma * X;
}

//2. 添加高斯噪声（加性噪声）
//k表示高斯噪声系数，k越大，高斯噪声系数越强
Mat AddGaussianNoise(const Mat &img, double mu, double sigma, int k) {
	Mat dst;
	dst.create(img.rows, img.cols, img.type());
	for (int x = 0; x < img.rows; x++) {
		for (int y = 0; y < img.cols; y++) {
			for (int c = 0; c < 3; c++) {
				double temp = img.at<Vec3b>(x, y)[c] + k * generateGaussianNoise(mu, sigma);
				if (temp > 255) {
					temp = 255;
				}
				else if (temp < 0) {
					temp = 0;
				}
				dst.at<Vec3b>(x, y)[c] = temp;
			}
		}
	}
	return dst;
}

//3.添加椒盐噪声，椒盐噪声是根据图像信噪比，随机生成一些图像内的像素位置并随机对这些像素点赋值为0或255
// SNR等于0-1的浮点数，用来控制选取位置的多少
Mat AddSaltNoise(const Mat &img, double SNR) {
	Mat dst;
	dst.create(img.rows, img.cols, img.type());
	int SP = img.rows * img.cols;
	int NP = SP*(1 - SNR); //获得需要添加椒盐噪声的像素个数
	dst = img.clone();
	for (int i = 0; i < NP; i++) {
		int x = (int)(rand()*1.0 / RAND_MAX * (double)img.rows);
		int y = (int)(rand()*1.0 / RAND_MAX * (double)img.cols);
		int r = rand() % 2;
		if (r) {
			dst.at<uchar>(x, y) = 0;
		}
		else {
			dst.at<uchar>(x, y) = 255;
		}
	}
	return dst;
}

//4. 调整图像饱和度(PhotoShop中的饱和度调节)
//(1) 计算每个像素点三基色最小值和最大值
//(2) delta为2值之差/255，如果二值之差为0不操作
//(3)value为两值之和
//(4)RGB图像空间转化为HSL(H色调,S饱和度,L亮度)
//L = value/2
//如果L<0.5，则S=delta/value
//否则S=delta/(2-value)
//Increment为饱和度,正值为增加饱和度,负值为降低饱和度,取值为(-1,1)
//(5)根据不同的公式得到新的rgb值
Mat ChangeColor(const Mat &img, const float Increment) {
	Mat dst;
	Mat Img_out(img.size(), CV_32FC3);
	img.convertTo(Img_out, CV_32FC3);
	Mat Img_in(img.size(), CV_32FC3);
	img.convertTo(Img_in, CV_32FC3);
	//定义输入图像的迭代器
	MatIterator_<Vec3f>inp_begin, inp_end;
	inp_begin = Img_in.begin<Vec3f>();
	inp_end = Img_in.end<Vec3f>();
	//定义输出图像的迭代器
	MatIterator_<Vec3f>out_begin, out_end;
	out_begin = Img_out.begin<Vec3f>();
	out_end = Img_out.end<Vec3f>();
	float delta = 0;
	float minVal, maxVal, t1, t2, t3, L, S, alpha;
	for (; inp_begin != inp_end; inp_begin++, out_begin++) {
		t1 = (*inp_begin)[0];
		t2 = (*inp_begin)[1];
		t3 = (*inp_begin)[2];

		minVal = std::min(std::min(t1, t2), t3);
		maxVal = std::max(std::max(t1, t2), t3);

		delta = (maxVal - minVal) / 255.0;
		L = 0.5 * (maxVal + minVal) / 255.0;
		S = std::max(0.5 * delta / L, 0.5 * delta / (1 - L));
		if (Increment > 0) {
			alpha = max(S, 1 - Increment);
			alpha = 1.0 / alpha - 1;
			(*out_begin)[0] = (*inp_begin)[0] + ((*inp_begin)[0] - L*255.0) * alpha;
			(*out_begin)[1] = (*inp_begin)[1] + ((*inp_begin)[1] - L*255.0) * alpha;
			(*out_begin)[2] = (*inp_begin)[2] + ((*inp_begin)[2] - L*255.0) * alpha;
		}
		else {
			alpha = Increment;
			(*out_begin)[0] = L*255.0 + ((*inp_begin)[0] - L*255.0) * (1 + alpha);
			(*out_begin)[1] = L*255.0 + ((*inp_begin)[1] - L*255.0) * (1 + alpha);
			(*out_begin)[2] = L*255.0 + ((*inp_begin)[2] - L*255.0) * (1 + alpha);
		}
	}
	Img_out.convertTo(dst, CV_8UC3);
	return dst;
}

Mat OldPicture(const Mat &src) {
	Mat Image_out(src.size(), CV_32FC3);
	src.convertTo(Image_out, CV_32FC3);
	Mat Image_2(src.size(), CV_32FC3);
	src.convertTo(Image_2, CV_32FC3);
	Mat r(src.rows, src.cols, CV_32FC1);
	Mat g(src.rows, src.cols, CV_32FC1);
	Mat b(src.rows, src.cols, CV_32FC1);
	Mat out[] = { b, g, r };
	split(Image_2, out);
	Mat r_new(src.rows, src.cols, CV_32FC1);
	Mat g_new(src.rows, src.cols, CV_32FC1);
	Mat b_new(src.rows, src.cols, CV_32FC1);
	r_new = 0.393*r + 0.769*g + 0.189*b;
	g_new = 0.349*r + 0.686*g + 0.168*b;
	b_new = 0.272*r + 0.534*g + 0.131*b;
	Mat rgb[] = { b_new, g_new, r_new };
	merge(rgb, 3, Image_out);
	Mat dstImg;
	Image_out.convertTo(dstImg, CV_8UC3);
	return dstImg;
}

Mat Scale(const Mat &src, double scale) {
	Mat dst;
	resize(src, dst, Size(src.cols*scale, src.rows*scale));
	return dst;
}

Mat Light(const Mat &src, int belta) {
	int rows = src.rows;
	int cols = src.cols;
	Mat dst(src.rows, src.cols, CV_8UC3);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < 3; k++) {
				dst.at<Vec3b>(i, j)[k] = src.at<Vec3b>(i, j)[k] + belta > 255 ? 255 : src.at<Vec3b>(i, j)[k] + belta;
			}
		}
	}
	return dst;
}

Mat Contrast(const Mat &src, double alpha) {
	int rows = src.rows;
	int cols = src.cols;
	Mat dst(src.rows, src.cols, CV_8UC3);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < 3; k++) {
				dst.at<Vec3b>(i, j)[k] = src.at<Vec3b>(i, j)[k] * alpha > 255 ? 255 : src.at<Vec3b>(i, j)[k] * alpha;
			}
		}
	}
	return dst;
}

//美白加磨皮
Mat SkinWhitening(const Mat &src) {
	int rows = src.rows, cols = src.cols;
	Mat HighPass(src.rows, src.cols, CV_8UC3);
	bilateralFilter(src, HighPass, 15, 100, 5); //PS中只需要做到这一步就好
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < 3; k++) {
				HighPass.at<Vec3b>(i, j)[k] = HighPass.at<Vec3b>(i, j)[k] - src.at<Vec3b>(i, j)[k] + 128;
				if (HighPass.at<Vec3b>(i, j)[k] < 0) HighPass.at<Vec3b>(i, j)[k] = 0;
				else if (HighPass.at<Vec3b>(i, j)[k] > 255) HighPass.at<Vec3b>(i, j)[k] = 255;
			}
		}
	}
	GaussianBlur(HighPass, HighPass, Size(1, 1), 0, 0);
	int Opacity = 60;
	Mat dst(src.rows, src.cols, CV_8UC3);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < 3; k++) {
				dst.at<Vec3b>(i, j)[k] = (src.at<Vec3b>(i, j)[k] * (100 - Opacity) + (src.at<Vec3b>(i, j)[k] + 2 * HighPass.at<Vec3b>(i, j)[k] - 256) * Opacity) / 100;
				if (dst.at<Vec3b>(i, j)[k] < 0) dst.at<Vec3b>(i, j)[k] = 0;
				else if (dst.at<Vec3b>(i, j)[k] > 255) dst.at<Vec3b>(i, j)[k] = 255;
			}
		}
	}
	return dst;
}

//偏色校正
//根据用户指定的R、G、B三个色彩的调整分量，分别附加到对应的色彩分量上，从而改变原始图像的色彩
Mat ColorBalance(const Mat &src) {
	int low = 150, high = 200;
	int R = rand() % (high - low + 1) + low;
	low = 10, high = 20;
	int G = rand() % (high - low + 1) + low;
	low = 5, high = 10;
	int B = rand() % (high - low + 1) + low;
	if (R < -255) {
		R = -255;
	}
	if(G < -255) {
		G = -255;
	}
	if (B < -255) {
		B = -255;
	}
	if (R > 255) {
		R = 255;
	}
	if (G > 255) {
		G = 255;
	}
	if (B > 255) {
		B = 255;
	}
	int rows = src.rows;
	int cols = src.cols;
	Mat dst(rows, cols, CV_8UC3);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			dst.at<Vec3b>(i, j)[0] = uchar(src.at<Vec3b>(i, j)[0] + B);
			dst.at<Vec3b>(i, j)[1] = uchar(src.at<Vec3b>(i, j)[1] + G);
			dst.at<Vec3b>(i, j)[2] = uchar(src.at<Vec3b>(i, j)[2] + R);
			if (dst.at<Vec3b>(i, j)[0] < 0) dst.at<Vec3b>(i, j)[0] = 0;
			else if (dst.at<Vec3b>(i, j)[0] > 255) dst.at<Vec3b>(i, j)[0] = 255;
			if (dst.at<Vec3b>(i, j)[1] < 0) dst.at<Vec3b>(i, j)[1] = 0;
			else if (dst.at<Vec3b>(i, j)[1] > 255) dst.at<Vec3b>(i, j)[1] = 255;
			if (dst.at<Vec3b>(i, j)[2] < 0) dst.at<Vec3b>(i, j)[2] = 0;
			else if (dst.at<Vec3b>(i, j)[2] > 255) dst.at<Vec3b>(i, j)[2] = 255;
		}
	}
	return dst;
}

//同态滤波
Mat HomoFilter(const Mat &src, int t = 1) {
	double t2 = (double)(t - 10) / 110;
	vector<Mat> rgb_split;
	cv::split(src, rgb_split);
	Mat dst;
	for (int i = 0; i < 3; i++)
	{
		Mat original = rgb_split[i].clone();
		Mat frame_log, padded, fourier_src, spatial, reinforce_src;
		original.convertTo(frame_log, CV_32FC1);
		frame_log += 1;
		log(frame_log, frame_log);
		//将图片从空域中转至频域
		int m = frame_log.rows;
		int n = frame_log.cols;
		copyMakeBorder(frame_log, padded, 0, m - frame_log.rows, 0, n - frame_log.cols, BORDER_CONSTANT, Scalar::all(0));
		Mat image_planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
		cv::merge(image_planes, 2, fourier_src);
		dft(fourier_src, fourier_src);


		//构造同态滤波器
		Mat hu(fourier_src.size(), CV_32FC1, Scalar::all(0));
		Point center = Point(hu.rows / 2, hu.cols / 2);
		for (int i = 0; i < hu.rows; i++)
		{
			float* data = hu.ptr<float>(i);
			for (int j = 0; j < hu.cols; j++)
				data[j] = (double)1 / (1 + pow(sqrt(pow(center.x - i, 2) + pow((center.y - j), 2)), -t2));
		}
		Mat butterworth_channels[] = { Mat_<float>(hu), Mat::zeros(hu.size(), CV_32F) };
		cv::merge(butterworth_channels, 2, hu);


		//进行频域卷积操作，得到强化过后的频域图，将其转回空域
		cv::mulSpectrums(fourier_src, hu, fourier_src, 0);
		cv::idft(fourier_src, spatial, DFT_SCALE);

		//对图像进行还原
		cv::exp(spatial, spatial);
		vector<Mat> planes;
		cv::split(spatial, planes);
		cv::magnitude(planes[0], planes[1], reinforce_src);
		/*Mat temp;
		normalize(reinforce_src, temp, 0, 255, NORM_MINMAX);
		temp.convertTo(temp, CV_8UC1);*/
		//这里采用偏差法对图像进行还原，类似的也可以用直方图归一化，只不过我试过直方图归一化效果不是很好
		Mat mean_val, stddev_value;
		cv::meanStdDev(reinforce_src, mean_val, stddev_value);
		double min, max, minmax;
		min = mean_val.at<double>(0, 0) - 2 * stddev_value.at<double>(0, 0);
		max = mean_val.at<double>(0, 0) + 2 * stddev_value.at<double>(0, 0);
		minmax = max - min;
		for (int i = 0; i < planes[0].rows; i++)
			for (int j = 0; j < planes[0].cols; j++)
				reinforce_src.at<float>(i, j) = 255 * (reinforce_src.at<float>(i, j) - min) / minmax;
		reinforce_src.convertTo(reinforce_src, CV_8UC1);
		rgb_split[i] = reinforce_src.clone();

	}
	cv::merge(rgb_split, dst);
	//下面对图像进行饱和度拉伸以及灰度线性变换以获得更加生动的图片
	cvtColor(dst, dst, COLOR_BGR2HSV);
	vector<Mat> hsv;
	split(dst, hsv);
	for (int i = 0; i < hsv[1].rows; i++)
		for (int j = 0; j < hsv[1].cols; j++)
			hsv[1].at<uchar>(i, j) = hsv[1].at<uchar>(i, j) * 4 / 3;
	for (int i = 0; i < hsv[1].rows; i++)
		for (int j = 0; j < hsv[1].cols; j++)
		{
			if (hsv[2].at<uchar>(i, j) < 235)
				hsv[2].at<uchar>(i, j) = hsv[2].at<uchar>(i, j) * 45 / 50 - 40;
		}
	merge(hsv, dst);
	cvtColor(dst, dst, COLOR_HSV2BGR);
	return dst;
}

//灰度化
Mat Gray(const Mat &src) {
	int rows = src.rows;
	int cols = src.cols;
	Mat dst(rows, cols, CV_8UC1);
	for (int i = 0; i < rows; i++) {
		uchar *ptrGray = dst.ptr<uchar>(i);
		const Vec3b *ptrRgb = src.ptr<Vec3b>(i);
		for (int j = 0; j < cols; j++) {
			ptrGray[j] = 0.3 * ptrRgb[j][2] + 0.59 * ptrRgb[j][1] + 0.11 * ptrRgb[j][0];
		}
	}
	return dst;
}

//轮换通道
Mat RotateChannel(const Mat &src) {
	int rows = src.rows;
	int cols = src.cols;
	Mat dst(rows, cols, CV_8UC3);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			dst.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[1];
			dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[2];
			dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[0];
		}
	}
	return dst;
}

//图像错切
//flag=1，水平错切; flag=-1, 垂直错切
//水平方向: x2=(x1-y1*tan(theta))
//         y2=y1
const double PI = 3.1415926;
Mat Slant(const Mat &src, float angle, int flag) {
	int rows = src.rows;
	int cols = src.cols;
	float ftan = fabs((float)tan(angle / 180.0*PI));
	int newHeight = 0;
	int newWidth = 0;
	if (flag == 1) { //水平方向高度不变
		newHeight = rows;
		newWidth = (int)(cols + rows*fabs(ftan));
	}
	else {//垂直方向宽度不变
		newHeight = (int)(rows + cols*fabs(ftan));
		newWidth = cols;
	}
	Mat dst(rows, cols, CV_8UC3);
	for (int i = 0; i < newHeight; i++) {
		for (int j = 0; j < newWidth; j++) {
			int newi, newj;
			if (flag == 1) {
				newi = i;
				newj = j + ftan * (i - rows);
			}
			else {
				newi = i + ftan * (j - cols);
				newj = j;
			}
			if (newi >= 0 && newi < rows && newj >= 0 && newj < cols) {
				for (int k = 0; k < 3; k++) {
					dst.at<Vec3b>(i, j)[k] = src.at<Vec3b>(newi, newj)[k];
				}
			}
			/*else {
				for (int k = 0; k < 3; k++) {
					dst.at<Vec3b>(i, j)[k] = 255;
				}
			}*/
		}
	}
	return dst;
}

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

//钝化蒙版
//degree: 钝化度，取值(0-100)
//钝化度用来改变像素之间的对比度强弱，钝化值越小，锐化的部分就越窄，仅仅会影响边缘的像素
//锐化值越大，锐化的范围越宽，效果越明显
Mat UnsharpMask(const Mat &src, int degree) {
	int rows = src.rows;
	int cols = src.cols;
	Mat dst;
	src.copyTo(dst);
	for (int i = 0; i < degree; i++) {
		GaussianBlur(dst, dst, Size(3, 3), 1.0);
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < 3; k++) {
				dst.at<Vec3b>(i, j)[k] = 2 * src.at<Vec3b>(i, j)[k] - dst.at<Vec3b>(i, j)[k];
			}
		}
	}
	return dst;
}