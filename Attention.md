# Attention  
---
## 本质  
Attention 是一种加权求和的方法，给在分类模型中提供更多贡献的图片部分较大的比重，贡献小的部分较小的比重，使得学习机能把更多的有效计算放在更合适的位置上，这样可以有效的减少计算量。  

![8CWqI0.png](https://s2.ax1x.com/2020/03/10/8CWqI0.png)

其中alpha就是权重。而Attention所做的事情就是把alpha计算出来。  
## 如何设计  
1.  根据hi和关注对象的相关关系设计一个打分函数。  
2.  对所得到的所有分数进行softmax得到最总的权重。  

# 论文阅读总结  
---
## Attention is all you need 
### Attention  
一个Attention Function被描述为对value向量做权重和计算，而权重是由query和key向量计算得来的。在传统的Attention中，q是来自外部的。  
### self-attention  
1. Scaled Dot-Product Attention  

![8CfCZR.png](https://s2.ax1x.com/2020/03/10/8CfCZR.png)

其中，Q,K,V分别是query,key,value的向量集合——矩阵。
2. Multi-Head Attention  

![8CfPd1.png](https://s2.ax1x.com/2020/03/10/8CfPd1.png)

其中W是学习机需要学习的参数。  
其本质就是利用多个Q进行Attention平行计算，使得每个Attention关注不同的部分以后再进行concat。  
![89Qq29.png](https://s2.ax1x.com/2020/03/09/89Qq29.png)

## Non-local Neural Network  
以前的网络都是一次处理一个局部的邻域，在这篇论文中，作者提出了一个非局部的操作来捕获远距离的像素与其关系。其实质就是把所有位置特征的加权和作为某一位置的响应，类似于Attention计算。
### Advantage of using Non-local operation  
+ 直接计算两个任意位置的相互作用来捕获两点的依赖性
+ 高效性，只有几层就可以达到最佳效果
+ 可变输入性，可以很容易的结合其他操作
### Formulation  
通用公式

![8CfeQe.png](https://s2.ax1x.com/2020/03/10/8CfeQe.png)

其中i是输出的像素点位置，j是枚举的所有像素点。x是输入的图片或者特征，y是输出的信息，其尺寸和x一样大（可变输入性）。  
f函数和g函数的其他变种：  
+ Gaussian  

![8CfrWT.png](https://s2.ax1x.com/2020/03/10/8CfrWT.png)

令归一化元素c(x)为f遍历j的和。  
+ Embedded Gaussian  

![8Cf4Fx.png](https://s2.ax1x.com/2020/03/10/8Cf4Fx.png)

令归一化元素c(x)为f遍历j的和。  
我们注意到最近提出来的应用于机器翻译的self-attention是在高斯嵌入函数的non-local operation的一个特例。因为归一化计算元素c(x)可以被看成是沿着j方向做的softmax运算。  
所以我们就得到了：  

![8CfTSO.png](https://s2.ax1x.com/2020/03/10/8CfTSO.png)

因此作者认为softmax是不必要的，为了说明其不必要性，作者还提出了以下两个选择；  
+ Dot product  

![8CfO0A.png](https://s2.ax1x.com/2020/03/10/8CfO0A.png)

+ Concatenation  

![8CfxtP.png](https://s2.ax1x.com/2020/03/10/8CfxtP.png)

### non-local block  
作者将上面的方程进行包装，得到一个block，从而可以方便地插入到现有的框架中去，一个局部块定义如下：  

![8Chk0s.png](https://s2.ax1x.com/2020/03/10/8Chk0s.png)

其中加xi是一个残差运算，是为了让其更好的插入到现有的网络中使用。  
具体操作如下图：  
![3xVnHI.png](https://s2.ax1x.com/2020/03/08/3xVnHI.png)  

## Residual Attention Network for Image Classification  
作者结合了Attention可以聚焦于特定区域计算的能力和residual net可以加深网络的特点，企图多次迭代Attention网络进行计算。
### properties  
+ 添加了更多的注意力机制，提升了网络性能，基于不同深度的特征图可以提取不同的注意力。
+ 残差网络可以结合目前的大多数已存网络，并且使用该策略可以轻易地将网络扩展到数百层。
### contributions  
1. 设计了一种可堆叠的网络结构，并且引入了注意力机制。  
2. 残差学习法，可以消除梯度消失现象，使得注意力模块可以得到充分学习。  
3. bottom-up与top-down结构相结合，自底向上是为了图片的特征提取，自顶向下是为了生成Attention map.  
### Residual Attention Network  
包含两部分：mask attention and trunk branch。  
主干部分的作用是特征提取，可以选择任意一种最先进的网络结构。掩膜分支通过对特征图的处理输出维度一致的注意力特征图（Attention Feature Map），然后使用点乘操作将两个分支的特征图组合在一起，得到最终的输出特征图。  

![8ChMX4.png](https://s2.ax1x.com/2020/03/10/8ChMX4.png)

其中T是主干部分的输出，M是掩膜部分的输出。i是空间所有的位置，c是channel的索引。  
增加残差结构以后变为:  

![8Ch37R.png](https://s2.ax1x.com/2020/03/10/8Ch37R.png)

其网络结构如下图所示：
![3xlvid.png](https://s2.ax1x.com/2020/03/08/3xlvid.png)  

### soft mask branch  
本结构之后包含快速前馈扫描和自顶向下的反馈步骤两个部分，前者可以快速的获取全图的信息，后者可以将将获得的全局信息与图片特征联系起来。这两个步骤反映了卷积网络上就是bottom-up和top-down结构（pooling and interpolation）。
![3x39h9.png](https://s2.ax1x.com/2020/03/08/3x39h9.png)  

### Spatial Attention and Channel Attention 
对于不同的Attention模型，采取了不同的归一化方程；对mixed Attention采用正常的sigmoid方程进行归一化，对于channels Attention采用了二范数的归一化，spacial Attention先对每一个通道的feature map进行了归一化，然后再进行sigmoid。

![8ChYh6.png](https://s2.ax1x.com/2020/03/10/8ChYh6.png)

其中mean和std分别代表了平局值和标准差。 

## Attention to Scale: Scale-aware Semantic Image Segmentation  
本文提出了一种注意力机制，在每一个像素点处对多尺度的特征进行柔性加权。注意力模型不仅优于取平均和最大池化，而且允许我们在不同的位置和尺度上诊断性地可视化特征的重要性。  
与以往把attention应用在2d的空间维度或者时间维度上的工作不同，我们探究了attention在尺寸维度上的作用。  
首先我们把一个最先进的语义分割模型用到分享网络中，然后采用soft-attention模型来代替平均和最大池化来对多尺寸输入的图片所得到的特征进行融合——对每一个尺寸的图片，attention都会生成一个weight map（为每一个像素给出权重），然后与经过FCNs处理后得到的Score map进行点乘。然后把所有尺寸的结果加起来再进行分类，具体如下图所示：  
![3x5Cc9.png](https://s2.ax1x.com/2020/03/08/3x5Cc9.png)  
### Attention model for scales  
用attention替换pooling之后，可以可视化每个图像位置在每个尺度上的特征的影响。  
其公式如下：  

![8Ch64P.png](https://s2.ax1x.com/2020/03/10/8Ch64P.png)

其中s指不同尺寸大小的图片，f是指score map的运算，c指有多少类关注对象，g是对f进行权重求和运算后的结果，在进行权重求和之前要先对f进行上采样获得一样的尺寸。  
权重w的计算如下:    

![8ChRgS.png](https://s2.ax1x.com/2020/03/10/8ChRgS.png)

其中h是指由attention计算得到的在尺寸为s的图片中位置i的score map，且w是所有通道共享的参数。  
![3z3UJJ.png](https://s2.ax1x.com/2020/03/08/3z3UJJ.png)  

## The Application of Two-level Attention Models in Deep Convolutional Neural Network for Fine-grained Image Classification  
在这篇论文中，作者将vision attention机制运用到fine-grained classification中。作者一共运用了三种attention：  
+ the bottom-up attention（提供备选patch）  
+ the object-level top-down attention（挑选出与目标相关的patch）  
+ the part-level top-down attention（定位出局部的区别）  
重要的是，作者避免使用了边界框或端到端的部分信息等昂贵的注释。  
###  Introduction  
大部分的fine-grained classification系统都是采取了两步走的策略，先获取图片中的object然后在根据细节区分种类。  
用bottom-up attention attention可以获取大量的小patch，但是patch过于粗糙，此时需要用top-down attention models来进行过滤没用的噪声，从而获取有用的patch。该过滤用的网络被称作Filternet，然后用挑选出来的图片训练一个网络，该网络被称作Domainnet,并用该网络进行细粒度分类。  
### object-level Attention model  
目的是去除噪声和与目标无关的像素，我们将一个在1K-class ILSVR2012 dataset训练好了的CNN网络作为filternet进行过滤。然后在训练出Domainnet网络，该网络有两个优点，一个是可以分类，另一个是其隐藏层具有聚类效果。  
object-level top-down attention如下图所示：  
![3zdT0A.png](https://s2.ax1x.com/2020/03/08/3zdT0A.png)  
### Part-Level Attention Model  
由于Domainnet的隐藏层具有聚类效果，比如，其一部分神经元负责鸟的头部，一部分负责身体，还有一部分负责翅膀，每一个聚类都代表了一个part detector。  
part-level top-down attention如下图所示：  
![3zwbDJ.png](https://s2.ax1x.com/2020/03/08/3zwbDJ.png) 
### Complete pipeline：  
[![3zyWE4.png](https://s2.ax1x.com/2020/03/08/3zyWE4.png)](https://imgchr.com/i/3zyWE4)  
## Learning Multi-Attention Convolutional Neural Network for Fine-Grained Image Recognition  
与上篇论文相同，细粒度分类问题分为两步走策略。但是目前主流的方法是将二者分开来分别计算，但是本文作者提出多注意卷积神经网络（MA-CNN），让part generation 和 feature learning能互相强化。
### Introduction  
作者发现了part generation 和 feature learning能互相强化，比如一开始的头部定位可以确定一个特定模式的学习，然后反过来加强头部的正确定位。因此作者提出了一种基于多注意力卷积神经网络(MA-CNN)的part-learning方法，该方法不需要边界框/part标注就可以进行细粒度识别。  
MA-CNN讨论了卷积、信道分组和局部分类子网，以全图像为输入，生成多个局部建议。
![89QcCQ.png](https://s2.ax1x.com/2020/03/09/89QcCQ.png)

### Approach  
首先，整个网络以下图 (a)中的全尺寸图像作为输入，将其送入下(b)中的卷积层中，提取基于区域的特征表示。其次，通过对(d)中的信道分组和加权层，生成 (e)中的多个部分注意图，然后使用sigmoid函数生成概率。其结果部分表征是由具有空间注意机制的基于区域的特征表征汇聚而成，如(f)所示。每一组概率得分/每一部分细粒度分类预测由全连通层和softmax层(g)得到,我们提出的MA-CNN通过交替学习每个部件表示上的softmax损失和每个部件注意图上的信道分组损失来优化，使其收敛。  
![8SMsfA.png](https://s2.ax1x.com/2020/03/09/8SMsfA.png)  
### Multi-Attention CNN for Part Localization   
![8ChTNq.png](https://s2.ax1x.com/2020/03/10/8ChTNq.png)

其中，W * X是指对输入图片X进行卷积网络处理，W是卷积网络参数。FC 层 f(.)输入的是特征，输出的每一个channel的权重向量。  

## Look Closer to See Better: Recurrent Attention Convolutional Neural Network for Fine-grained Image Recognition  
本文提出了一个全新的循环注意力卷积神经网络（recurrent attention convolutional neural network——RA-CNN），用互相强化的方式对判别区域注意力（discriminative region attention）和基于区域的特征表征（region-based feature representation）进行递归学习。该算法在每一个尺度的图片训练中都包含一个分类子网络和一个注意力建议子网络（APN）,APN开始于一个完整的图片，然后由粗到精，每一次迭代都参考了上次的预测结果。  
RA-CNN 通过尺度内分类损失（intra-scale classification loss）和尺度间排序损失（inter-scale ranking loss）进行优化。  
![8Sro4A.png](https://s2.ax1x.com/2020/03/09/8Sro4A.png)  
### Approach  
作者在网络结构设计了3个scale子网络（也可以继续叠加），每个scale子网络的网络结构都是一样的（保证了关注对象不变），只是网络参数不一样，下一个scale的输入是上一个scale的关注对象，在每个scale子网络中包含两种类型的网络：分类网络和APN网络。因此数据流是这样的：输入图像通过分类网络提取特征并进行分类，然后APN网络基于提取到的特征进行训练得到attention region信息，再将attention region剪裁（crop）出来并放大（zoom in），再作为第二个scale网络的输入。  
![8ScHIg.png](https://s2.ax1x.com/2020/03/09/8ScHIg.png)  
### Attention Proposal Network  
首先我们定义如下公式为提取图片特征的卷积预处理，W为卷积层参数,X为输入图片；

![8ChLgU.png](https://s2.ax1x.com/2020/03/10/8ChLgU.png)

我们让我们的模型在每一个尺度输出两个结果，第一个结果是输出一个针对所有的子类的概率分布P，f函数代表的全连接层和一个softmax；

![8C4VDH.png](https://s2.ax1x.com/2020/03/10/8C4VDH.png)

另一个输出为关注对象的边界盒子的坐标及大小；g函数就是APN网络，下标x,y是坐标，下标l是正方形边长的一半；

![8C4nUI.png](https://s2.ax1x.com/2020/03/10/8C4nUI.png)

根据输出的坐标及边长我们就可以对注意的目标进行定位于放大。
### Loss Function  
损失函数分为两部分：inter-scale pairwise ranking loss和pairwise ranking loss

![8C41xS.png](https://s2.ax1x.com/2020/03/10/8C41xS.png)

其中s代表的是尺度，损失函数的前半部分是尺度内分类损失，Y(s)是不同尺度的图片输出的标签向量，Y * 是真的标签向量，他优化的是卷积层和优化层的参数。第二部分是尺度间排序损失；

![8C4Nan.png](https://s2.ax1x.com/2020/03/10/8C4Nan.png)

这种设计可以使网络以粗糙尺度的预测为参考，通过加强精细尺度的网络工作来逐步逼近最具鉴别性的区域，从而产生更可靠的预测。  

## An Empirical Study of Spatial Attention Mechanisms in Deep Networks  
本文将Transformer attention和deformable convo- lution和dynamic convolution modules中所应用的self-attention进行试验比较后，发现了一些特点。比如query and keycontent在Transformer中的作用非常小，但是在encode-decode中的作用却比较大。另一方面，合理的结合deformable convolution和key content可以得到更好的结果。  
### Introduction Attention  
Attention模型在自然语言处理上的成功使得人们想把它运用到计算机视觉上，然后Transformer的变种就被运用到了识别工作中，比如object detection和semantic segmentation，在这里，query和key就是指图片像素或者感兴趣的区域。  
决定注意力权重的因素通常就三个；query content，key content 和query与key的相对位置。  
对于self-attention的情况，query content可以是图像中查询像素处的特征，也可以是一个句子中的一个单词的特征。key是query的邻域像素，或者句子中的另个单词。  
因此在学习一个key相对于query的权重时，可以分出四个重要因素：  
+ query,key的内容  
+ query的内容和相对位置  
+ key的内容  
+ 相对位置  
注意力权重就可以写成以上四项的和:  

![8C4ose.png](https://s2.ax1x.com/2020/03/10/8C4ose.png)

![8pZkhq.png](https://s2.ax1x.com/2020/03/09/8pZkhq.png)  
在论文中，作者提出三个重要发现，

1. 在Transformer的检测模块中，查询敏感词，特别是查询词和关键内容词在self-attention中起着次要的作用。但在编码器和解码器的关注中，查询和关键字是至关重要的。
2. 虽然可变形卷积技术仅基于查询查询和相对位置项来实现注意机制，但它在图像识别中比变形卷积技术更有效。  
3. 在self-attention中，查询内容、相对位置和关键内容是最重要的因素。
并且，作者提出了一个观点，注意力模块的设计相对于self-attention的固有特征更加决定学习机最终的结果。  
### Study of Spatial Attention Mechanisms  
我们开发了一个广义的attention公式化，能够代表各种模块的设计。
以下是multi-head attention feature计算公式：

![8C5Pds.png](https://s2.ax1x.com/2020/03/10/8C5Pds.png)

其中q是query element(Zq)的索引，k是key element(Xk)的索引，m是head数目，W是参数。  
#### Transformer attention  
![8C5tyD.png](https://s2.ax1x.com/2020/03/10/8C5tyD.png)

其中；

![8C57lT.png](https://s2.ax1x.com/2020/03/10/8C57lT.png)

这两项对query content比较敏感，其中U,V是可以学习的embedding matrices，Rk-q是通过计算将相对位置映射到高维空间以后的结果。  

![8C5bXF.png](https://s2.ax1x.com/2020/03/10/8C5bXF.png)

这两项对query content无关，v,u是可以学习的向量。  

## Look and Think Twice: Capturing Top-Down Visual Attention with Feedback Convolutional Neural Networks∗  
人类在看一张图片的时候，可能第一眼看过去也看不到某些信息，但是根据第一次看到的结果，再仔细看的时候，就能发现一些明显的隐藏信息在第一眼的时候被忽略了，而神经网络也是一样，在传递的过程中也会忽略一些信息，而增加反馈机制，就能提高网络解决实际分类和定位的问题的有效性。  
视觉注意力是指人们在大脑中先定位一个目标然后再带着这个目的去图中寻找，这是一种反馈机制，是一种自上而下的注意机制。  
![8SwCI1.png](https://s2.ax1x.com/2020/03/09/8SwCI1.png)  