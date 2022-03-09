# tianchi-AI-Challenger-sea8

天池 AAAI2022安全AI挑战者计划第八期

## 赛事简介

当前的机器学习竞赛主要是基于固定的数据集去追求一个高性能的机器学习模型，而最近的以数据为中心的人工智能竞赛 (https://https-deeplearning-ai.github.io/data-centric-comp) 改变了传统范式，即给定固定模型旨在去改进数据集。类似地，在鲁棒学习方面，已经提出了基于深度学习模型的防御方法来减轻对抗样本的潜在威胁，但大多数方法都是在固定的约束和数据集下去追求高性能模型。因此，目前尚未广泛探索如何构建通用且有效的数据集来训练鲁棒模型。

在图像分类的对抗鲁棒性研究中，为了加快以数据为中心的相关的技术研究，我们组织了本次比赛，目的是开发新的以数据为中心的算法，例如数据增强、标签细化、制造对抗性数据，甚至设计来自其他领域的知识融合算法。鼓励参与者自由开发新颖的想法，找到有效的以数据为中心的技术，以促进训练更加鲁棒的机器学习模型。

## 模型和数据集

模型：ResNet50，DenseNet121

数据集：CIFAR10

## 硬件和软件

设备：2080Ti

系统：Ubuntu 18.0

深度学习框架：Pytorch1.6

## 思路

#### 思考过程

这个比赛的训练代码是不可以修改的，可以修改的只有`config.py`和训练数据。

> 小吐槽：前期查了不少关于data-centric AI的资料，后来发现这个比赛就是拼对抗样本的。

**baseline：**只使用CIFAR10的1w张test样本进行训练的话只有78分左右（需要调整config文件中的参数）

**扩大数据来源：**尝试使用CIFAR10所有的6w个样本（从中随机抽取5w张），只有60多分，结合群里大家的讨论，应该是线上测试集是或者大部分是CIFAR10的测试集通过各种篡改得到的。

**随机数据增强：**对1W张样本加上高斯噪声，生成额外的1w张样本，分数提升到将近90，生成额外4w张，分数提升到92分。后来尝试使用imgaug库进行更多样的数据增强，一共可以到93+，再尝试数据增强我觉得就没有意义了。

**对抗样本：**对抗样本指对输入样本故意添加一些人无法觉察的细微干扰，导致模型以高置信度给出一个错误的输出。由于我之前在网上看到过对抗样本的有关报导，还挺感兴趣的，所以大概试了两天数据增强达到瓶颈后就想到了用对抗样本来做，发现确实是一个行之有效的办法，后面的尝试也都是围绕对抗样本来进行。

- FGSM，这个是最经典的对抗样本生成方法，尝试自己写了一版，1w原始数据+3w随机数据增强+1wFGSM在$l_2$约束下产生的对抗样本，分数可以到95+
- 尝试更多的对抗样本生成方式，通过对抗样本生成库[foolbox]([bethgelab/foolbox: A Python toolbox to create adversarial examples that fool neural networks in PyTorch, TensorFlow, and JAX (github.com)](https://github.com/bethgelab/foolbox)) 基于FGSM、PGD、Deepfool、PonitwiseAttack等方式生成对抗样本。

#### **最终方案：**

原数据1w+随机数据增强1w+对抗样本3w（FGSM和PGD在$l_2$和$l_\infty$范数约束），分数达到97.43

#### 其他方案：

也尝试了一些我认为可以提升模型鲁棒性，但实际上分数并没有提高的方案：

1. 模拟adversarial training

   adversarial training在训练模型的时候，根据模型当前的参数生成对抗样本，作为模型的输入（详见`adversarial_training.py`）使得模型更加鲁棒

   由于这个比赛固定了训练的pipeline，所以不能使用adversarial training的方式来提升模型的鲁棒性。但是我写了一个adversarial training的代码，保存每次生成的对抗样本，最后随机抽取5w张。

2. 防御蒸馏

   知识蒸馏是原先hinton提出用来减少模型复杂度并且不会降低泛化性能的方法，具体就是在指定温度下，先训练一个教师模型，再将教师模型在数据集上输出的类别概率标记作为软标签训练学生模型。而在防御蒸馏模型中，选择两个相同的模型作为教师模型和学生模型。

   硬标签训练教师模型，然后用教师模型输出的软标签训练学生模型。

3. 用GAN来做数据增强

   我队友做了，训练GAN很麻烦，而且效果还不如添加高斯噪声

4. soft label

   - $L_i=\alpha, L_j=\frac{\alpha}{K-1}$
   - 训练一个更大的模型来生成伪标签

5. Mixup数据增强

## 最终成绩

队伍名称：大家好

score:97.43

排名81/3691

## 感想

第一次打比赛，天池的这个比赛对新人比较友好，不需要配置很复杂的环境，而且一张2080Ti足够我训练模型，有一个能让我快速上手的baseline，给我增加了不少信心。打比赛能快速学到很多知识，钉钉群里也结交了很多大佬。

不得不说排行榜还是有点卷，最后我们大约再提升个0.3就能进决赛了，可是97分以上再提升一点就很难。赛后看了决赛圈大佬的方法，发现和我的相差无几，最后可能就是差点运气，还是有点遗憾。

## TOP方案总结

// TODO

## Reference

- [vtddggg/training_template_for_AI_challenger_sea8: Example code of [Tianchi AAAI2022 Security AI Challenger Program Phase 8\] (github.com)](https://github.com/vtddggg/training_template_for_AI_challenger_sea8)
- [bethgelab/foolbox: A Python toolbox to create adversarial examples that fool neural networks in PyTorch, TensorFlow, and JAX (github.com)](https://github.com/bethgelab/foolbox)

