# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 22:02:01 2022

@author: qiang
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 01:10:47 2021

@author: qiang
"""
#import sys
#sys.path.append('../')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import metrics as mc
import pandas as pd
#import MLNN_Dataprepare as dpa

import Model_Metrics as mmc



A=np.load("gdata/Majiang_training.npy")
[Train,validt]=A.item(),A.item()
#[Train,validt]=[np.load('gdata/Data_train_ML.npy'),0]#Train: 训练数据集, Test: 测试数据集
X =Train["data"].reshape(Train["data"].shape[0],Train["data"].shape[1]*Train["data"].shape[2])
Lb =Train["label"]
y=np.zeros(len(Lb[:,0]))
for i in range(Lb.shape[1]):y=y+i*Lb[:,i]

'''
A=np.load('gdata/pressing_letters.npy')
X=A[:,:1024]/255
adds=np.random.uniform(-0.2,0.2,X.shape)
X=X+adds
y=A[:,1024]
'''


 
#y=np.log2((4*Train[:,-1]+2*Train[:,-2]+Train[:,-3]))




def pca_for_all_selected_classses(yls):
    
    pca = PCA(n_components=6)
    X_p = pca.fit(X).transform(X)
    np.save('gdata/Data_PCA.npy',X_p)
    print(X_p.shape)
    #mc.GPCR_wells[30:36]
    #targetname=mc.GPCR_wells[0:6]
    #mc.labels_means('plate1',conds=targetname,labels=mc.plate1_labels)
    ax = plt.figure(figsize=(4,4))

    ax=plt.subplot(1,1,1)  
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    ax.spines['bottom'].set_linewidth(1);##设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1);##设置左边坐标轴的粗细
    
    lgs=np.arange(len(set(y)))
    
    
    for m,c,i,target_name in zip(mc.markers,mc.colors,lgs,yls):
        print(target_name)
        plt.scatter(X_p[y==i,0], X_p[y==i, 1],c='w',edgecolors=c,marker=m,label=target_name,s=5)

    plt.xticks([-8,-6,-4,-2,0,2,4,6,8],fontsize=10,rotation=0,fontweight='bold')
    plt.yticks(fontsize=10,rotation=0,fontweight='bold')
    plt.xlim(-8,8)
    plt.ylim(-4,4)
    
    #plt.xlabel('PC1',fontsize=10,fontweight='bold')
    #plt.ylabel('PC2',fontsize=10,fontweight='bold')
    #plt.legend(loc=1,ncol=1,fontsize=10)
    plt.legend(bbox_to_anchor=(1.25, 1),loc=1,ncol=1,fontsize=10)
     
    
    #plt.title('Cluster')
    
    plt.savefig('saved_figs/pca_first_at.png',bbox_inches='tight', dpi=300)
    plt.show()



tlabels=mc.yls

pca_for_all_selected_classses(tlabels)