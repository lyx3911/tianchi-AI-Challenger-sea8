'''
Author: your name
Date: 2021-11-24 14:26:36
LastEditTime: 2021-12-13 09:45:41
LastEditors: Please set LastEditors
FilePath: /training_template_for_AI_challenger_sea8/data_vis/data_vis.py
'''
import numpy as np 
import cv2

if __name__ == '__main__':
    number_pics = np.load("attack_data/val-baseline_FGSM_multi-eps_data_densenet121.npy")[:100]
    labels = np.load("attack_data/val-baseline_FGSM_multi-eps_label_densenet121.npy")[:100]
    # print(labels)
    # new_labels = labels
    # for i in range(labels.shape[0]): 
    #     for j in range(labels.shape[1]):
    #         if labels[i,j] > 0:
    #             new_labels[i,j] = 1.0
    #             # print(labels[i,j], i, j)
    #         elif labels[i,j] < 0: 
    #             print(labels[i,j], i, j)
    # np.save("./new_labels_1.npy", new_labels)
    # print(new_labels)
    
    data_len = number_pics.shape[0]
    labels = np.argmax(labels,axis=1)
    unique, counts = np.unique(labels, return_counts=True)
    print(unique, counts)
    for index in range(data_len):
        number_pic = number_pics[index]
        # print(number_pic)        
        label = np.argmax(labels[index])
        # print(label)
        cv2.imwrite("data_vis/FGSM-multi-epsilon/{}.jpg".format(index), number_pic)
        # break
         