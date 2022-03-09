'''
Author: your name
Date: 2021-11-24 14:26:36
LastEditTime: 2022-01-10 19:33:27
LastEditors: Please set LastEditors
FilePath: /training_template_for_AI_challenger_sea8/data_vis/data_vis.py
'''
import numpy as np 
import cv2

if __name__ == '__main__':
    number_pics = np.load("save/pgd_l2_1.5_attack_data/attack_data_densenet121.npy")[:10000]
    print(number_pics.shape)
    labels = np.load("save/pgd_l2_1.5_attack_data/attack_label_densenet121.npy")[:10000]
    
    data_len = number_pics.shape[0]
    labels = np.argmax(labels,axis=1)
    unique, counts = np.unique(labels, return_counts=True)
    print(unique, counts)    
    
    for epoch in range(0,data_len//100):
        epoch_data = np.zeros([320,320,3], dtype = np.uint8)
        # print(epoch_data.shape)
        for i in range(10): 
            for j in range(10):
                data = number_pics[epoch*100+i*10+j]
                epoch_data[32*i:32*(i+1), 32*j:32*(j+1), :] = data
        cv2.imwrite("save/pgd_l2_1.5_attack_data/data_vis/epoch-{}.jpg".format(epoch), epoch_data)