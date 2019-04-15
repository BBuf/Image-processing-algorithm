vector <double> getW(double coor, double a) {
	vector <double> w(4);
	int base = static_cast<int>(coor); //取整作为基准
	double e = coor - static_cast<double>(base); //多出基准的小数部分
	vector <double> tmp(4);//存放公式中 |x| <= 1,  1 < |x| < 2四个值
	//4 x 4的16个点，所以tmp[0]和tmp[4]距离较远，值在[1, 2]区间
	tmp[0] = 1.0 + e;//1 < x < 2
	tmp[1] = e;// x <= 1
	tmp[2] = 1.0 - e; // x <= 1
	tmp[3] = 2.0 - e; // 1 < x < 2
	//按照cubic的公式计算系数w
	//x <= 1
	w[1] = (a + 2.0) * std::abs(std::pow(tmp[1], 3)) - (a + 3.0) * std::abs(std::pow(tmp[1], 2)) + 1;
	w[2] = (a + 2.0) * std::abs(std::pow(tmp[2], 3)) - (a + 3.0) * std::abs(std::pow(tmp[2], 2)) + 1;
	// 1 < x < 2
	w[0] = a * std::abs(std::pow(tmp[0], 3)) - 5.0 * a * std::abs(std::pow(tmp[0], 2)) + 8.0*a*std::abs(tmp[0]) - 4.0*a;
	w[3] = a * std::abs(std::pow(tmp[3], 3)) - 5.0 * a * std::abs(std::pow(tmp[3], 2)) + 8.0*a*std::abs(tmp[3]) - 4.0*a;
	return w;
}

Mat BicubicInterpolation(Mat src, float sx, float sy) {
	int row = src.rows;
	int col = src.cols;
	int channels = src.channels();
	int dst_row = round(row * sx);
	int dst_col = round(col * sy);
	sx = 1.0 / sx;
	sy = 1.0 / sy;
	Mat dst(dst_row, dst_col, CV_8UC1, Scalar::all(0));
	for (int i = 2; i < dst_row - 2; i++) {
		for (int j = 2; j < dst_col - 2; j++) {
			double r = static_cast <double> (i * sx);
			double c = static_cast <double> (j * sy);
			if (r < 1.0) r += 1.0;
			if (c < 1.0) c += 1.0;
			vector <double> a = getW(r, -0.5);
			vector <double> b = getW(c, -0.5);
			double sum = 0;
			int cc = static_cast<int>(c);
			int rr = static_cast<int>(r);
			if (cc > col - 3) {
				cc = col - 3;
			}
			if (rr > row - 3) {
				rr = row - 3;
			}
			std::vector<std::vector<int> > src_arr = {
				{ src.at<uchar>(rr - 1, cc - 1), src.at<uchar>(rr, cc - 1), src.at<uchar>(rr + 1, cc - 1), src.at<uchar>(rr + 2, cc - 1) },
				{ src.at<uchar>(rr - 1, cc), src.at<uchar>(rr, cc), src.at<uchar>(rr + 1, cc), src.at<uchar>(rr + 2, cc) },
				{ src.at<uchar>(rr - 1, cc + 1), src.at<uchar>(rr, cc + 1), src.at<uchar>(rr + 1, cc + 1), src.at<uchar>(rr + 2, cc + 1) },
				{ src.at<uchar>(rr - 1, cc + 2), src.at<uchar>(rr, cc + 2), src.at<uchar>(rr + 1, cc + 2), src.at<uchar>(rr + 2, cc + 2) }
			};
			for (int k = 0; k < 3; k++) {

			}
			for (int x = 0; x < 3; x++) {
				for (int y = 0; y < 3; y++) {
					sum += a[x] * b[y] * static_cast<double>(src_arr[x][y]);
				}
			}
			dst.at<uchar>(i, j) = static_cast<int>(sum);
		}
	}
	return dst;
}
