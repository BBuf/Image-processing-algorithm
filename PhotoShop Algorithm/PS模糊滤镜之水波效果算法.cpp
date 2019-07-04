const double pi = acos(-1.0);
Mat WaveFilter(Mat src) {
	int row = src.rows;
	int col = src.cols;
	float A = 7;
	float B = 2.5;
	Point Center(col / 2, row / 2);
	Mat dst(row, col, CV_8UC3);
	for (int y = 0; y < row; y++) {
		for (int x = 0; x < col; x++) {
			float y0 = Center.y - y;
			float x0 = x - Center.x;
			float theta = atan(y0 / (x0 + 0.00001));
			if (x0 < 0)
				theta = theta + pi;
			float r0 = sqrt(x0*x0 + y0*y0);
			float r1 = r0 + A * col * 0.01 * sin(B * 0.1 * r0);
			
			float new_x = r1 * cos(theta);
			float new_y = r1 * sin(theta);

			new_x = Center.x + new_x;
			new_y = Center.y + new_y;

			if (new_x < 0) new_x = 0;
			if (new_x >= col - 1) new_x = col - 2;
			if (new_y < 0) new_y = 0;
			if (new_y >= row - 1) new_y = row - 2;
			int x1 = (int)new_x;
			int y1 = (int)new_y;
			float p = new_x - x1;
			float q = new_y - y1;
			for (int k = 0; k < 3; k++) {
				dst.at<Vec3b>(y, x)[k] = (1 - p)*(1 - q)*src.at<Vec3b>(y1, x1)[k] + (p)*(1 - q)*src.at<Vec3b>(y1, x1 + 1)[k] + 
					(1 - p)*(q)*src.at<Vec3b>(y1 + 1, x1)[k] + (p)*(q)*src.at<Vec3b>(y1 + 1, x1 + 1)[k];
			}
		}
	}
	return dst;
}