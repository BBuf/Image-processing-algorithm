double Transform(double x)
{
	if (x <= 0.05)return x * 2.64;
	return 1.099*pow(x, 0.9 / 2.2) - 0.099;
}

struct zxy {
	double x, y, z;
}s[2500][2500];

int work(cv::Mat input_img, cv::Mat out_img) {
	int rows = input_img.rows;
	int cols = input_img.cols;
	double r, g, b;
	double lwmax = -1.0, base = 0.75;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			b = (double)input_img.at<Vec3b>(i, j)[0] / 255.0;
			g = (double)input_img.at<Vec3b>(i, j)[1] / 255.0;
			r = (double)input_img.at<Vec3b>(i, j)[2] / 255.0;
			s[i][j].x = (0.4124*r + 0.3576*g + 0.1805*b);
			s[i][j].y = (0.2126*r + 0.7152*g + 0.0722*b);
			s[i][j].z = (0.0193*r + 0.1192*g + 0.9505*b);
			lwmax = max(lwmax, s[i][j].y);
		}
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			double xx = s[i][j].x / (s[i][j].x + s[i][j].y + s[i][j].z);
			double yy = s[i][j].y / (s[i][j].x + s[i][j].y + s[i][j].z);
			double tp = s[i][j].y;
			//修改CIE:X,Y,Z
			s[i][j].y = 1.0 * log(s[i][j].y + 1) / log(2 + 8.0*pow((s[i][j].y / lwmax), log(base) / log(0.5))) / log10(lwmax + 1);
			double x = s[i][j].y / yy*xx;
			double y = s[i][j].y;
			double z = s[i][j].y / yy*(1 - xx - yy);

			//转化为用RGB表示
			r = 3.2410*x - 1.5374*y - 0.4986*z;
			g = -0.9692*x + 1.8760*y + 0.0416*z;
			b = 0.0556*x - 0.2040*y + 1.0570*z;

			if (r < 0)r = 0; if (r>1)r = 1;
			if (g < 0)g = 0; if (g>1)g = 1;
			if (b < 0)b = 0; if (b>1)b = 1;

			//修正补偿
			r = Transform(r), g = Transform(g), b = Transform(b);
			out_img.at<Vec3b>(i, j)[0] = int(b * 255);
			out_img.at<Vec3b>(i, j)[1] = int(g * 255);
			out_img.at<Vec3b>(i, j)[2] = int(r * 255);
		}
	}
	//cv::imshow("result", out_img);
	//cv::imwrite("E:\\NCNN_work\\result.jpg", out_img);
	//waitKey(0);
	return 0;
}
