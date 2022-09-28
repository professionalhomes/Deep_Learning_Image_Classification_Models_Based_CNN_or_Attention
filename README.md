# Deep-Learning-Models-For-Classification
This project organizes classic classification Neural Networks based  convolution or attention mechanism.

This readme is introduced in Chinese (including most of the comments in the code). Please translate it into English if necessary.

## 1. Introduction

以下是本项目支持的模型列表，包含了自AelxNet以来经典的深度学习分类模型，大部分模型是基于卷积神经网络的，也有一部分是基于注意力机制的。  博客链接是对模型的介绍，会持续更新...

在项目目录中，模型的搭建代码在classic_models文件夹中；**所有的模型训练代码是共用的**，有三个版本：
- train_sample.py是最简单的实现，必须掌握，以下版本看个人能力和需求。
- train.py是升级版的实现，具体改进的地方见train.py脚本中的注释。
- train_distrubuted.py支持多gpu分布式训练。  

最后，test.py是推理脚本，用于使用训练好的模型。dataload中是数据集加载代码；utils是封装的功能包，包括学习策略，训练和验证，分布式初始化，可视化等等。建议先学习掌握classic_models，train_sample.py和test.py这三部分。

## 2. Dataset And Project 
本项目是使用python语言基于pytorch深度学习框架编写的。

默认的数据集是花朵数据集，此数据集包含五种不同种类的花朵图像，用于训练的图像有3306张，用于验证的图像有364张。下载链接如下：https://pan.baidu.com/s/1EhPMVLOQlLNN55ndrLbh4Q 
提取码：7799 。

**下载完成后，记得在训练和推理代码中，将数据集加载的路径修改成自己电脑中下载存储的路径。**

数据集图像展示如下： 
![Dataset show](https://user-images.githubusercontent.com/102544244/192847344-958812cc-0988-4fa4-a458-ed842c41b8d2.png)


开启模型的训练只需要在IDE中执行train_sample.py脚本即可；或者在终端执行命令行`python train_sample.py` 训练的log打印示例如下：
![training log](https://user-images.githubusercontent.com/102544244/192849338-d7297768-88d4-40f8-83b6-79962ace7fd4.png)

将训练好的模型用于推理，给一张向日葵的图像，模型的输出结果示例结果如下：
![infer show](https://user-images.githubusercontent.com/102544244/192850216-f9ebf217-97f9-4c87-a5e5-4c1e032f436b.png)

## 3. Methods And Papers
1. AlexNet        
Blog Introduction Link: [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E4%B8%80alexnet)
2. ZFNet          
Blog Introduction Link: WRINTING

3. VggNet  
Blog Introduction Link: https://blog.csdn.net/qq_39297053/article/details/123716634  

4. GoogleNet  
Blog Introduction Link: https://blog.csdn.net/qq_39297053/article/details/123717625  

5. ResNet  
Blog Introduction Link: https://blog.csdn.net/qq_39297053/article/details/123739792  

6. DenseNet  
Blog Introduction Link: https://blog.csdn.net/qq_39297053/article/details/123765554  

7. MobileNet  
Blog Introduction Link: https://blog.csdn.net/qq_39297053/article/details/123793236  

8. ShuffleNet  
Blog Introduction Link: https://blog.csdn.net/qq_39297053/article/details/123797686  

9. SENet  
Blog Introduction Link: https://blog.csdn.net/qq_39297053/article/details/123848298  

10. Vision_Transformer  
Blog Introduction Link: WRINTING  

11. Swin_Transformer  
Blog Introduction Link: WRINTING  

12. EfficientNet  
Blog Introduction Link: https://blog.csdn.net/qq_39297053/article/details/123804502  

13. ConvNeXt  
Blog Introduction Link: WRINTIN  

14. MLP-mixer  
Blog Introduction Link: WRINTING... ...  

In additon, I write training and inference python scripts for image classification task.
train.py 

## configures
本项目是使用python语言基于pytorch深度学习框架编写的。
此外，我写了三个训练脚本用于模型的训练，默认的数据集是花朵数据集，此数据集包含五种不同种类共三千多张花朵图像，下载链接：链接：https://pan.baidu.com/s/1EhPMVLOQlLNN55ndrLbh4Q 
提取码：7799 。如要使用，请指定参超到数据集地址/flower（eg： --data_path /.../.../.../flower）

三个训练脚本中，train_sample.py是最简单的实现；train.py是升级版的实现，具体改进点见train.py脚本中的注释; train_distrubuted.py支持多gpu分布式训练。  

最后，test.py是推理脚本。dataload中是数据集加载代码；utils是封装的功能包，包括学习策略，训练和验证，分布式初始化，可视化等等。
