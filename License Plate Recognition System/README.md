# 车牌号码识别算法

- LP.cpp 利用OpenCV实现了对车牌区域的检测，判断，切割，以及调用Python脚本进行预测字符
- train-license-digits.py 实现对英文和字数字字符识别模型进行训练
- train-license-province.py 实现对汉字字符识别模型进行训练
- estimate_chinese.py 利用汉字字符识别模型对省份字符进行预测
- estimate_character.py 利用英文和数字字符识别模型对英文和数字进行预测