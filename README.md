该项目主要实现卷积神经网络Lenet-5 的训练（Python+Kreas+Jupyter Notebook）和推理C语言实现(C+Visual Studio2013)
项目特点：卷积的参数（输入宽、输入高、输入通道数、输出通道数、卷积核的大小、步幅大小、pad（1是进行补边,0是不补边）,激活函数（1是relu,0是softmax）可调，通用性比较强。全连接使用卷积运算实现、池化采用最大池化、SMAE padding。

工程的实现主要分为两部分：
第一部分是在python训练Lenet-5-MNIST模型
0.环境：Python+Kreas+Jupyter Notebook
1.Lenet-5网络搭建模型，训练、保存模型。     （train.ipynb）
2.手写字体预测。通过画图工具获得手写字体，并进行模型推理得到预测结果。  （test.ipynb）
3.提取网络模型每一层的参数（权重和bias），保存为.bin文件。   （get_layers_weight.ipynb）
4.提取测试图片在网络每一层中 的输出结果。（便于在后续c语言实现中对比结果，验证程序的正确性）（get_layer_output.ipynb）
5.将图片转为.bin格式，包括图片的宽高大小、通道数和像素值（image2array.py）

第二部分工作是网络前向推理C实现
0.环境：C+Visual Studio2013
1.主函数main.cpp，主函数类似于python中模型搭建，在主函数中调用各个层的的函数，完成向前传播的过程。
2.minst.cpp，主要定义了图片和标签结构体、图片读取，权重和偏置读取函数。
3.cnn.cpp，定义了卷积层，池化层，flatten层函数，以便在主函数中调用。
4.根据需要修改网络结构，以及数据存放位置。
5.直接运行就可以了



