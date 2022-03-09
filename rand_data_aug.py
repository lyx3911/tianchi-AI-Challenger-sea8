import numpy as np
import cv2
from imgaug import augmenters as iaa
import imgaug as ia
import tqdm
import random

data = np.load("data.npy")
label = np.load("label.npy")
soft_label = label

print(data.shape, label.shape)

# s = random.sample(range(0, 60000), 25000)
# # print(s)
# data = data[s]
# label = label[s]
# print(data.shape, label.shape)

sometimes = lambda aug: iaa.Sometimes(0.5, aug)


seq = iaa.Sequential(
    [
        # 加入高斯噪声
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.05*255), per_channel=0.5
            ),
        # 使用下面的0个到5个之间的方法去增强图像。注意SomeOf的用法
    #     iaa.SomeOf((2, 3),
    #     [
    #         #用高斯模糊，均值模糊，中值模糊中的一种增强。注意OneOf的用法
    #         iaa.OneOf([
    #             iaa.GaussianBlur((0, 3.0)),
    #             iaa.AverageBlur(k=(2, 7)), # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
    #             iaa.MedianBlur(k=(3, 11)),
    #         ]),

    #         # #锐化处理
    #         # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

    #         # #浮雕效果
    #         # iaa.Emboss(alpha=(0, 1.0), strength=(0, 1.1)),

    #         # #边缘检测，将检测到的赋值0或者255然后叠在原图上
    #         # sometimes(iaa.OneOf([
    #         #     iaa.EdgeDetect(alpha=(0, 0.7)),
    #         #     iaa.DirectedEdgeDetect(
    #         #         alpha=(0, 0.7), direction=(0.0, 1.0)
    #         #     ),
    #         # ])),

    #         # 加入高斯噪声
    #         iaa.AdditiveGaussianNoise(
    #             loc=0, scale=(0.0, 0.05*255), per_channel=0.5
    #         ),

    #         # 将1%到10%的像素设置为黑色
    #         # 或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖
    #         iaa.OneOf([
    #             iaa.Dropout((0.01, 0.1), per_channel=0.5),
    #             iaa.CoarseDropout(
    #                 (0.03, 0.15), size_percent=(0.02, 0.05),
    #                 per_channel=0.2
    #             ),
    #         ]),

    #         # #5%的概率反转像素的强度，即原来的强度为v那么现在的就是255-v
    #         # iaa.Invert(0.05, per_channel=True), 

    #         # 每个像素随机加减-10到10之间的数
    #         iaa.AddElementwise((-10,-10)),

    #         # 像素乘上0.5或者1.5之间的数字.
    #         iaa.Multiply((0.5, 1.5), per_channel=0.5),

    #         # # 将整个图像的对比度变为原来的一半或者二倍
    #         # iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),

    #         # # 将RGB变成灰度图然后乘alpha加在原图上
    #         # iaa.Grayscale(alpha=(0.0, 1.0)),
            
    #         # 椒盐噪声
    #         iaa.SaltAndPepper(0.1), 
    #     ],
            
    #         random_order=True # 随机的顺序把这些操作用在图像上
    #     )
    ],
    random_order=True # 随机的顺序把这些操作用在图像上
)

##### random data augmentations #################################
datalen = data.shape[0]

for aug_time in range(1):
    data_aug = []
    for i in tqdm.tqdm(range(datalen)):
        image = data[i]
        image_aug = seq.augment_image(image)
        data_aug.append(image_aug)
        # cv2.imwrite("./data_vis/data_aug/{}.jpg".format(i), image_aug)
    data = np.concatenate([data_aug, data])
    label = np.concatenate([label, soft_label])
data = data[:10000]
label = label[:10000]

attack_data1 = np.concatenate([np.load("save/pgd_linf_12_attackdata/attack_data_densenet121.npy"), np.load("save/pgd_linf_12_attackdata/attack_data_resnet50.npy")])
attack_label1 = np.concatenate([np.load("save/pgd_linf_12_attackdata/attack_label_densenet121.npy"), np.load("save/pgd_linf_12_attackdata/attack_label_resnet50.npy")])

attack_data2 = np.concatenate([np.load("save/pgd_l2_1.5_attack_data/attack_data_densenet121.npy"), np.load("save/pgd_l2_1.5_attack_data/attack_data_resnet50.npy")])
attack_label2 = np.concatenate([np.load("save/pgd_l2_1.5_attack_data/attack_label_densenet121.npy"), np.load("save/pgd_l2_1.5_attack_data/attack_label_resnet50.npy")])

# attack_data = np.load("./attack_data/foolbox-attack-baselinelr0.01-Linf0.1-L2-2_data_resnet50.npy")
# attack_label = np.load("./attack_data/foolbox-attack-baselinelr0.01-Linf0.1-L2-2_label_resnet50.npy")

attack_data = np.concatenate([attack_data1, attack_data2])
attack_label = np.concatenate([attack_label1, attack_label2])

print(attack_data.shape, attack_label.shape)
# labels = np.argmax(attack_label,axis=1)
# unique, counts = np.unique(labels, return_counts=True)
# print(unique, counts)

s = random.sample(range(0, attack_data.shape[0]), 40000)
data = np.concatenate([data, attack_data[s]])
label = np.concatenate([label, attack_label[s]])

print(data.shape, label.shape)

np.save("commit/gauss-LinfPGD-12-L2PGD-1.5-0o1r4a/data.npy", data)
np.save("commit/gauss-LinfPGD-12-L2PGD-1.5-0o1r4a/label.npy", label)
