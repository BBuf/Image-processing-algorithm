//	方案（2）对应的代码，速度稍慢
//void GetGrayIntegralImage(unsigned char *Src, int *Integral, int Width, int Height, int Stride)
//{
//	int *ColSum = (int *)calloc(Width, sizeof(int));		//	用的calloc函数哦，自动内存清0
//	memset(Integral, 0, (Width + 1) * sizeof(int));
//	for (int Y = 0; Y < Height; Y++)
//	{
//		unsigned char *LinePS = Src + Y * Stride;
//		int *LinePL = Integral + Y * (Width + 1) + 1;
//		int *LinePD = Integral + (Y + 1) * (Width + 1) + 1;
//		LinePD[-1] = 0;
//		for (int X = 0; X < Width; X++)
//		{
//			ColSum[X] += LinePS[X];
//			LinePD[X] = LinePD[X - 1] + ColSum[X];
//		}
//	}
//	free(ColSum);
//}


void GetGrayIntegralImage(unsigned char *Src, int *Integral, int Width, int Height, int Stride)
{
	//	你所看到的每一句代码都是作者辛勤劳作和多年经验的积累，希望你能尊重作者的成果
	//	你的每一个  评论  和  打赏  都是作者撰写更多博文和分享经验的鼓励。
	//	本代码对应博文见：http://www.cnblogs.com/Imageshop/p/6219990.html
	
	memset(Integral, 0, (Width + 1) * sizeof(int));					//	第一行都为0
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *LinePS = Src + Y * Stride;
		int *LinePL = Integral + Y * (Width + 1) + 1;				//	上一行位置			
		int *LinePD = Integral + (Y + 1) * (Width + 1) + 1;			//	当前位置，注意每行的第一列的值都为0
		LinePD[-1] = 0;												//	第一列的值为0
		for (int X = 0, Sum = 0; X < Width; X++)
		{
			Sum += LinePS[X];										//	行方向累加
			LinePD[X] = LinePL[X] + Sum;							//	更新积分图
		}
	}
}

void GetRGBIntegralImage(unsigned char *Src, int *Integral, int Width, int Height, int Stride)
{
	//	你所看到的每一句代码都是作者辛勤劳作和多年经验的积累，希望你能尊重作者的成果
	//	你的每一个  评论  和  打赏  都是作者撰写更多博文和分享经验的鼓励。
	//	本代码对应博文见：http://www.cnblogs.com/Imageshop/p/6219990.html

	const int Channel = 3;
	memset(Integral, 0, (Width + 1) * Channel * sizeof(int));
	for (int Y = 0; Y < Height; Y++)
	{
		unsigned char *LinePS = Src + Y * Stride;
		int *LinePL = Integral + Y * (Width + 1) * Channel + Channel;
		int *LinePD = Integral + (Y + 1) * (Width + 1) * Channel + Channel;
		LinePD[-3] = 0; LinePD[-2] = 0; LinePD[-1] = 0;
		for (int X = 0, SumB = 0, SumG = 0, SumR = 0; X < Width; X++)
		{
			SumB += LinePS[0];
			SumG += LinePS[1];
			SumR += LinePS[2];
			LinePD[0] = LinePL[0] + SumB;
			LinePD[1] = LinePL[1] + SumG;
			LinePD[2] = LinePL[2] + SumR;
			LinePS += Channel;
			LinePL += Channel;
			LinePD += Channel;
		}
	}
}



void BoxBlur(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride, int Radius)
{
	//	你所看到的每一句代码都是作者辛勤劳作和多年经验的积累，希望你能尊重作者的成果
	//	你的每一个  评论  和  打赏  都是作者撰写更多博文和分享经验的鼓励。
	//	本代码对应博文见：http://www.cnblogs.com/Imageshop/p/6219990.html
	
	int Channel = Stride / Width;
	int *Integral = (int *)malloc((Width + 1) * (Height + 1) * Channel * sizeof(int));
	if (Channel == 1)
	{
		GetGrayIntegralImage(Src, Integral, Width, Height, Stride);
		#pragma omp parallel for
		for (int Y = 0; Y < Height; Y++)
		{
			int Y1 = max(Y - Radius, 0);
			int Y2 = min(Y + Radius + 1, Height);
			//	int Y1 = Y - Radius;
			//	int Y2 = Y + Radius + 1;
			//	if (Y1 < 0) Y1 = 0;
			//	if (Y2 > Height) Y2 = Height;
			int *LineP1 = Integral + Y1 * (Width + 1);
			int *LineP2 = Integral + Y2 * (Width + 1);
			unsigned char *LinePD = Dest + Y * Stride;
			for (int X = 0; X < Width; X++)
			{
				int X1 = max(X - Radius, 0);
				int X2 = min(X + Radius + 1, Width);
				//	int X1 = X - Radius;
				//	if (X1 < 0) X1 = 0;
				//	int X2 = X + Radius + 1;
				//	if (X2 > Width) X2 = Width;
				int Sum = LineP2[X2] - LineP1[X2] - LineP2[X1] + LineP1[X1];
				int PixelCount = (X2 - X1) * (Y2 - Y1);					//	有效的像素数
				LinePD[X] = (Sum + (PixelCount >> 1)) / PixelCount;		//	四舍五入
			}
		}
	}
	else if (Channel == 3)
	{
		GetRGBIntegralImage(Src, Integral, Width, Height, Stride);
		#pragma omp parallel for
		for (int Y = 0; Y < Height; Y++)
		{
			int Y1 = max(Y - Radius, 0);
			int Y2 = min(Y + Radius + 1, Height);
			int *LineP1 = Integral + Y1 * (Width + 1) * 3;
			int *LineP2 = Integral + Y2 * (Width + 1) * 3;
			unsigned char *LinePD = Dest + Y * Stride;
			for (int X = 0; X < Width; X++)
			{
				int X1 = max(X - Radius, 0);
				int X2 = min(X + Radius + 1, Width);
				int Index1 = X1 * 3;
				int Index2 = X2 * 3;
				int SumB = LineP2[Index2 + 0] - LineP1[Index2 + 0] - LineP2[Index1 + 0] + LineP1[Index1 + 0];
				int SumG = LineP2[Index2 + 1] - LineP1[Index2 + 1] - LineP2[Index1 + 1] + LineP1[Index1 + 1];
				int SumR = LineP2[Index2 + 2] - LineP1[Index2 + 2] - LineP2[Index1 + 2] + LineP1[Index1 + 2];
				int PixelCount = (X2 - X1) * (Y2 - Y1);
				LinePD[0] = (SumB + (PixelCount >> 1)) / PixelCount;
				LinePD[1] = (SumG + (PixelCount >> 1)) / PixelCount;
				LinePD[2] = (SumR + (PixelCount >> 1)) / PixelCount;
				LinePD += 3;
			}
		}
	}
	free(Integral);
}