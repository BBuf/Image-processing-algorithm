//1. 图像旋转
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
	}else{
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
			double temp = img.at<uchar>(x, y) + k * generateGaussianNoise(mu, sigma);
			if (temp > 255) {
				temp = 255;
			}
			else if (temp < 0) {
				temp = 0;
			}
			dst.at<uchar>(x, y) = temp;
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

//5. 给图片增加老照片效果
Mat OldPicture(const Mat &src) {
	Mat Image_out(src.size(), CV_32FC3);
	src.convertTo(Image_out, CV_32FC3);
	Mat Image_2(src.size(), CV_32FC3);
	src.convertTo(Image_2, CV_32FC3);
	Mat r(src.rows, src.cols, CV_32FC1);
	Mat g(src.rows, src.cols, CV_32FC1);
	Mat b(src.rows, src.cols, CV_32FC1);
	Mat out[] = {b, g, r};
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

