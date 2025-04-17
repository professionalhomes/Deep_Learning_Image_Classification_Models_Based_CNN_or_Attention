# Deep Learning Image Classification Models Based CNN or Attention
This project organizes classic Neural Networks based convolution or attention mechanism, in order to solve image classification.

This readme is introduced in Chinese (including most of the comments in the code). Please translate it into English if necessary.

## 1. Introduction

This project contains classic deep learning image classification models since AelxNet, most of the models are based on convolutional neural networks and some are based on attention mechanisms. The blog link is an introduction to the models, which will be continuously updated...

In the project directory, the model building code is in the classic_models folder; **All the model training code and inference code are shared, only the model building code is different**, there are three different versions of the training code: 
- train_sample.py is the simplest implementation, which must be mastered, and the following two versions depend on the individual needs.
- train.py is the simple upgraded implementation, see the comments at the top in the train.py script for specific improvements.
- train_distrubuted.py supports multi-gpu distributed training.  

Finally, test.py is the inference script for using the trained model. dataload is the dataset loading code; utils is the package that encapsulates various functions, including learning strategy, training and validation, distributed initialization, visualization, and so on. It is recommended to learn and master the three parts of classic_models, train_sample.py and test.py first, and then learn the other parts when they are used.
## 2. Dataset And Project 
This project is written in python language based on pytorch deep learning framework.

The default dataset is flower dataset, this dataset contains five different kinds of flower images, there are 3306 images for training and 364 images for validation. The download link is as follows: https://pan.baidu.com/s/1EhPMVLOQlLNN55ndrLbh4Q 
Extract code: 7799 .

** After the download is complete, remember to change the path where the dataset is loaded to the path where it is downloaded and stored in your own computer in the training and inference code. ** 

The dataset images are displayed below:

Translated with DeepL.com (free version)
<div align="center">
  <img src="https://user-images.githubusercontent.com/102544244/192847344-958812cc-0988-4fa4-a458-ed842c41b8d2.png"  alt="Dataset show" width="700"/>
</div>
  
 
Enabling training of the model is simply a matter of executing the train_sample.py script in the IDE; or executing the command line `python train_sample.py` in the terminal An example of a log printout of the training is shown below:
<div align="center">
  <img src="https://user-images.githubusercontent.com/102544244/192849338-d7297768-88d4-40f8-83b6-79962ace7fd4.png"  alt="training log" width="600"/>
</div>
 
Using the model for inference is simply a matter of executing the test.py script in the IDE; or executing the command line `python test.py` in the terminal Given an image of a sunflower, the output of the model Sample results are as follows:

<div align="center">
  <img src="https://user-images.githubusercontent.com/102544244/192850216-f9ebf217-97f9-4c87-a5e5-4c1e032f436b.png"  alt="infer show" width="400"/>
</div>
 

## 3. Methods And Papers
The following is a list of models supported by this program
1. **[AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)**
   - *ImageNet Classification with Deep Convolutional Neural Networks*

2. **[ZFNet](https://arxiv.org/abs/1311.2901)**
   - *Visualizing and Understanding Convolutional Networks*

3. **[VGGNet](https://arxiv.org/abs/1409.1556)**
   - *Very Deep Convolutional Networks for Large-Scale Image Recognition*

4. **[GoogleNet (Inception v1)](https://arxiv.org/abs/1409.4842)**
   - *Going Deeper with Convolutions*

5. **[ResNet](https://arxiv.org/abs/1512.03385)**
   - *Deep Residual Learning for Image Recognition*

6. **[DenseNet](https://arxiv.org/abs/1608.06993)**
   - *Densely Connected Convolutional Networks*

7. **[SENet](https://arxiv.org/abs/1709.01507)**
   - *Squeeze-and-Excitation Networks*

8. **[MobileNet](https://arxiv.org/abs/1704.04861)**
   - *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*

9. **[ShuffleNet](https://arxiv.org/abs/1707.01083)**
   - *ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices*

10. **[EfficientNet](https://arxiv.org/abs/1905.11946)**
    - *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*

11. **[RepVGG](https://arxiv.org/abs/2101.03697)**
    - *RepVGG: Making VGG-style ConvNets Great Again*

12. **[Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)**
    - *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*

13. **[Swin Transformer](https://arxiv.org/abs/2103.14030)**
    - *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*

14. **[Visual Attention Network](https://arxiv.org/abs/2202.09741)**
    - *Visual Attention Network*

15. **[ConvNeXt](https://arxiv.org/abs/2201.03545)**
    - *A ConvNet for the 2020s*

16. **[MLP-Mixer](https://arxiv.org/abs/2105.01601)**
    - *MLP-Mixer: An All-MLP Architecture for Vision*

17. **[AS-MLP](https://arxiv.org/abs/2107.08391)**
    - *AS-MLP: An Axial Shifted MLP Architecture for Vision*

18. **[ConvMixer](https://arxiv.org/abs/2201.09792)**
    - *Patches Are All You Need?*

19. **[MetaFormer](https://arxiv.org/abs/2111.11418)**
    - *MetaFormer is Actually What You Need for Vision*

---

