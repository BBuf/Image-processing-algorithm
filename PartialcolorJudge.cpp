void RGB2LAB(Mat &rgb, Mat &Lab) {
	//RGB 转XYZ
	Mat XYZ(rgb.size(), rgb.type());
	Mat_ <Vec3b>::iterator beginRGB = rgb.begin<Vec3b>();
	Mat_ <Vec3b>::iterator endRGB = rgb.end<Vec3b>();
	Mat_ <Vec3b>::iterator beginXYZ = XYZ.begin<Vec3b>();
	int shift = 22;
	for (; beginRGB != endRGB; beginRGB++, beginXYZ++)
	{
		(*beginXYZ)[0] = ((*beginRGB)[0] * 199049 + (*beginRGB)[1] * 394494 + (*beginRGB)[2] * 455033 + 524288) >> (shift - 2);
		(*beginXYZ)[1] = ((*beginRGB)[0] * 75675 + (*beginRGB)[1] * 749900 + (*beginRGB)[2] * 223002 + 524288) >> (shift - 2);
		(*beginXYZ)[2] = ((*beginRGB)[0] * 915161 + (*beginRGB)[1] * 114795 + (*beginRGB)[2] * 18621 + 524288) >> (shift - 2);
	}
	//XYZ转LAB
	int LabTab[1024];
	for (int i = 0; i < 1024; i++)
	{
		if (i>9)
			LabTab[i] = (int)(pow((float)i / 1020, 1.0F / 3) * (1 << shift) + 0.5);
		else
			LabTab[i] = (int)((29 * 29.0 * i / (6 * 6 * 3 * 1020) + 4.0 / 29) * (1 << shift) + 0.5);
	}
	const int ScaleLC = (int)(16 * 2.55 * (1 << shift) + 0.5);
	const int ScaleLT = (int)(116 * 2.55 + 0.5);
	const int HalfShiftValue = 524288;
	beginXYZ = XYZ.begin<Vec3b>();
	Mat_<Vec3b>::iterator endXYZ = XYZ.end<Vec3b>();
	Lab.create(rgb.size(), rgb.type());
	Mat_<Vec3b>::iterator beginLab = Lab.begin<Vec3b>();
	for (; beginXYZ != endXYZ; beginXYZ++, beginLab++)
	{
		int X = LabTab[(*beginXYZ)[0]];
		int Y = LabTab[(*beginXYZ)[1]];
		int Z = LabTab[(*beginXYZ)[2]];
		int L = ((ScaleLT * Y - ScaleLC + HalfShiftValue) >> shift);
		int A = ((500 * (X - Y) + HalfShiftValue) >> shift) + 128;
		int B = ((200 * (Y - Z) + HalfShiftValue) >> shift) + 128;
		(*beginLab)[0] = L;
		(*beginLab)[1] = A;
		(*beginLab)[2] = B;
	}
}

//true代表存在偏色，false代表不存在偏色
bool PartialcolorJudge(Mat &imgLab) {
	Mat_<Vec3b>::iterator begin = imgLab.begin<Vec3b>();
	Mat_<Vec3b>::iterator end = imgLab.end<Vec3b>();
	float suma = 0, sumb = 0;
	for (; begin != end; begin++) {
		suma += (*begin)[1];//a
		sumb += (*begin)[2];//b
	}
	int MN = imgLab.rows * imgLab.cols;
	double Da = suma / MN - 128; //归一化到[-128,127]
	double Db = sumb / MN - 128; //同上
	//求平均色度
	double D = sqrt(Da * Da + Db * Db);
	begin = imgLab.begin<Vec3b>();
	double Ma = 0, Mb = 0;
	//求色度中心距
	for (; begin != end; begin++) {
		Ma += abs((*begin)[1] - 128 - Da);
		Mb += abs((*begin)[2] - 128 - Db);
	}
	Ma = Ma / MN;
	Mb = Mb / MN;
	double M = sqrt(Ma * Ma + Mb * Mb);
	float K = float(D / M);
	if (K >= 1.5) {
		return true;
	}
	else {
		return false;
	}
}