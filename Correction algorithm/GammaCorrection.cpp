/*
来自：https://blog.csdn.net/linqianbi/article/details/78617615
gamma校正原理：
　　假设图像中有一个像素，值是 200 ，那么对这个像素进行校正必须执行如下步骤： 
　　1. 归一化 ：将像素值转换为  0 ～ 1  之间的实数。 算法如下 : ( i + 0. 5)/256  这里包含 1 个除法和 1 个加法操作。对于像素  A  而言  , 其对应的归一化值为  0. 783203 。 

　　2. 预补偿 ：根据公式  , 求出像素归一化后的 数据以  1 /gamma  为指数的对应值。这一步包含一个 求指数运算。若  gamma  值为  2. 2 ,  则  1 /gamma  为  0. 454545 , 对归一化后的  A  值进行预补偿的结果就 是  0. 783203 ^0. 454545 = 0. 894872 。 

　　3. 反归一化 ：将经过预补偿的实数值反变换为  0  ～  255  之间的整数值。具体算法为 : f*256 - 0. 5  此步骤包含一个乘法和一个减法运算。续前 例  , 将  A  的预补偿结果  0. 894872  代入上式  , 得到  A  预补偿后对应的像素值为  228 , 这个  228  就是最后送 入显示器的数据。

  
　　如上所述如果直接按公式编程的话，假设图像的分辨率为 800*600 ，对它进行 gamma 校正，需要执行 48 万个浮点数乘法、除法和指数运算。效率太低，根本达不到实时的效果。 
　　针对上述情况，提出了一种快速算法，如果能够确知图像的像素取值范围  , 例如  , 0 ～ 255 之间的整数  , 则图像中任何一个像素值只能 是  0  到  255  这  256  个整数中的某一个 ; 在  gamma 值 已知的情况下  ,0 ～ 255  之间的任一整数  , 经过“归一 化、预补偿、反归一化”操作后 , 所对应的结果是唯一的  , 并且也落在  0 ～ 255  这个范围内。
　　如前例  , 已知  gamma  值为  2. 2 , 像素  A  的原始值是  200 , 就可求得 经  gamma  校正后  A  对应的预补偿值为  228 。基于上述原理  , 我们只需为  0 ～ 255  之间的每个整数执行一次预补偿操作  , 将其对应的预补偿值存入一个预先建立的  gamma  校正查找表 (LUT:Look Up Table) , 就可以使用该表对任何像素值在  0 ～ 255  之 间的图像进行  gamma  校正。
*/
Mat gammaTransform(Mat &src, float kFactor){
    unsigned char LUT[256];
    for (int i = 0; i < 256; i++){
        float f = (i + 0.5f) / 255;
        f = (float)(pow(f, kFactor));
        LUT[i] = saturate_cast<uchar>(f*255.0f - 0.5f);
    }
    Mat dst = src.clone();
    if (src.channels() == 1){
        MatIterator_<uchar> iterator = dst.begin<uchar>();
        MatIterator_<uchar> iteratorEnd = dst.end<uchar>();
        for (; iterator != iteratorEnd; iterator++){
            *iterator = LUT[(*iterator)];
        }
    }else{
        MatIterator_<Vec3b> iterator = dst.begin<Vec3b>();
        MatIterator_<Vec3b> iteratorEnd = dst.end<Vec3b>();
        for (; iterator != iteratorEnd; iterator++){
            (*iterator)[0] = LUT[((*iterator)[0])];
            (*iterator)[1] = LUT[((*iterator)[1])];
            (*iterator)[2] = LUT[((*iterator)[2])];
        }
    }
    return dst;
}
int main()
{
    Mat src = imread("../tmp.jpg");
    //取两种不同的gamma值
    float gamma1 = 3.33f;
    float gamma2 = 0.33f;
    float kFactor1 = 1 / gamma1;
    float kFactor2 = 1 / gamma2;
    Mat result1 = gammaTransform(src, kFactor1);
    Mat result2 = gammaTransform(src, kFactor2);
    imshow("origin", src);
    imshow("result1", result1);
    imshow("result2", result2);
    waitKey(0);
    return 0;
}
