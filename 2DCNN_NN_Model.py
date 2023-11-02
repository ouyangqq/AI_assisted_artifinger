# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 21:16:20 2021
@author: qiang
"""
import sys
import time
import numpy as np
import pandas as pd 
import warnings
from torch.utils.data.sampler import WeightedRandomSampler
import torch
from torch import optim 
import random
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import metrics as mc

import Model_Metrics as mmc

#import Model_Metrics as mmc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DTs=np.load("gdata/Majiang_training.npy").item()
DTs=np.hstack([DTs["data"].reshape(len(DTs["data"]),1024),DTs["label"]])

for ch in range(10):
    A1=DTs[DTs[:,ch+1024]==1,:]
    rands=random.sample(range(0,len(A1)),len(A1)) 
    tmp1=A1[rands[0:int(len(A1)*0.5)],:]
    tmp2=A1[rands[int(len(A1)*0.5):],:]
    if(ch==0):Train,validt=tmp1,tmp2
    Train=np.vstack([Train,tmp1])
    validt=np.vstack([validt,tmp2])


#[Train,Test]=DTs,DTs
#Train: 训练数据集, Test: 测试数据集


X=torch.from_numpy(Train[:,0:1024].reshape(Train.shape[0],32,32).astype(np.float32)).to(device)
Y=torch.from_numpy(Train[:,1024:].astype(np.float32)).to(device)
classnum=Y.shape[1]

X_test=torch.from_numpy(validt[:,0:1024].reshape(validt.shape[0],32,32).astype(np.float32)).to(device)
Y_test=torch.from_numpy(validt[:,1024:].astype(np.float32)).to(device)

XX_test=X_test
YY_test=Y_test


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=7, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=7)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256,64)  # 必须为16*5*5
        self.fc2 = nn.Linear(64, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        in_size = x.size(0)
        #print("--------------------",in_size)
        out = self.relu(self.mp(self.conv1(x)))
        #print("--------------------",in_size)
        out = self.relu(self.mp(self.conv2(out)))        
        #out = self.relu(self.conv3(out))
        out = out.view(in_size, -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return self.logsoftmax(out)
   
epochs=200
alaph=1e-4
Batchsize=128
#net= Anomaly_Classifier(input_size=1,num_classes=Y.shape[1]).to(device)
net = Net().to(device)

      
def Train_model():  
    print("OPTIMIZER = optim.Adam(model.parameters(),lr = 0.001) \n ")
    print('Functions Ready')
    #定义损失函数 与 优化器0
    criterion = nn.MSELoss() 
    #criterion = nn.MSELoss()
    #criterion = nn.BCEWithLogitsLoss
    optimizer = optim.Adam(net.parameters(), lr=alaph)
    #optimizer = optim.SGD(model.parameters(), lr=alaph)
    #optimizer = optim.Adam(anom_classifier.parameters(),lr = 0.001) 
    start = time.time()
    loss_accbuf=[]
    acc1,acc2=0,0
    
    for epoch in range(epochs):
        selnum=Y.shape[0]
        rands=random.sample(range(0,selnum),selnum)  
        running_loss = 0
        step = 0
        t1=time.time()
        for i in range(int(selnum/Batchsize)+1):
            step += 1
            rand_sels = rands[Batchsize*i:Batchsize*(i+1)]
            #print(input_data.shape)
            buf=Y[rand_sels]
            #label=torch.from_numpy(buf.astype(np.float32))
            label = torch.reshape(buf,(buf.shape[0],1,buf.shape[1]))
            optimizer.zero_grad()
            buf=X[rand_sels,:,:]
            #input_data = torch.from_numpy(buf.astype(np.float32))
            x = torch.reshape(buf,(buf.shape[0],1,buf.shape[1],buf.shape[2]))
            
            #print("ddd",x.shape,x.size(0),label.shape)
            outputs = net(x)#x[:,:,0:28,0:28]
     
            outputs=torch.reshape(outputs,(outputs.shape[0],1,outputs.shape[1]))
    
            loss = criterion(outputs,label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            max_vals,Truths1= torch.max(label[:,0,0:classnum],1)
            acc1=mc.calc_accuracy(outputs[:,0,0:classnum],Truths1)
            
            '''
            max_vals,Truths2=torch.max(Y_test[:,0:classnum],1)
            buf=X_test
            test_outputs = net(torch.reshape(buf,(buf.shape[0],1,buf.shape[1])) )
            #print(Truths2.shape,test_outputs.shape)
            acc2=mc.calc_accuracy(test_outputs[:,0:classnum],Truths2)
            #acc2=0
            '''
        for i in range(int(Y_test.shape[0]/Batchsize)+1):
            buf=Y_test[Batchsize*i:Batchsize*(i+1)]
            label = torch.reshape(buf,(buf.shape[0],1,buf.shape[1]))
            buf=X_test[Batchsize*i:Batchsize*(i+1),:]
            x = torch.reshape(buf,(buf.shape[0],1,buf.shape[1],buf.shape[2]))
            outputs1 = net(x)
            if(i==0): final_y=outputs1.cpu().detach().numpy()
            else:final_y=np.vstack([final_y,outputs1.cpu().detach().numpy()])
            #torch.cat((final_y,outputs), 0)
             
        final_y=torch.from_numpy(final_y.astype(np.float32)).to(device)
        max_vals,Truths= torch.max(Y_test[:,0:classnum],1)
        acc2=mc.calc_accuracy(final_y[:,0:classnum],Truths)   
   
            
        loss_accbuf.append([running_loss-233.6,acc1,acc2])
        if (epoch % 1==0):
            t2=time.time()
            print('time: %0.3f'% (t2-t1))
            print('[%d, %5d] loss: %0.6f  training acc %0.6f, validating acc %0.6f' % (epoch + 1, i + 1, running_loss-233.6,acc1,acc2))
            running_loss = 0.0

    print('time = %2dm:%2ds' % ((time.time() - start)//60, (time.time()-start)%60))
    torch.save(net.state_dict(),'model_weights/loss_cnn_nn_%d' % epochs+'_'+str(alaph)+'.pth') 
    np.save('model_weights/loss_cnn_nn_%d' % epochs+'_'+str(alaph)+'.npy',np.array(loss_accbuf)) 

# Train_model()


lossaccbf=np.load('model_weights/loss_cnn_nn_%d' % epochs+'_'+str(alaph)+'.npy') 
net.load_state_dict(torch.load('model_weights/loss_cnn_nn_%d' % epochs+'_'+str(alaph)+'.pth')) 

X_test=XX_test
Y_test=YY_test
classnum=YY_test.shape[1]


for i in range(int(Y_test.shape[0]/Batchsize)+1):
    buf=Y_test[Batchsize*i:Batchsize*(i+1)]
    #print(i,buf.shape)
    label = torch.reshape(buf,(buf.shape[0],1,buf.shape[1]))
    buf=X_test[Batchsize*i:Batchsize*(i+1),:]
    x = torch.reshape(buf,(buf.shape[0],1,buf.shape[1],buf.shape[1]))
    outputs = net(x)
    if(i==0): final_y=outputs.cpu().detach().numpy()
    else:final_y=np.vstack([final_y,outputs.cpu().detach().numpy()])
    #torch.cat((final_y,outputs), 0)
    
final_y=torch.from_numpy(final_y.astype(np.float32)).to(device)
#N=1000
#buf = torch.reshape(X_test[:N,:],(N,1,500))
#final_y = net(buf)
max_vals,Truths= torch.max(Y_test[:,0:classnum],1)
res=mc.calc_accuracy(final_y[:,0:classnum],Truths)

print('Overall Accuracy',res)
orgLabel=Y_test[:,0:classnum]
org_pred=final_y[:,0:classnum]


mmc.plot_confusion_matrix(org_pred,orgLabel,sfname='CNN',classes=mc.yls)
mmc.plot_accuracy(lossaccbf,sfname='CNN')
mmc.plot_roc(Y_test,final_y,mc.yls,sfname='CNN')  


hpfile=r'../../Human_perception.xlsx'
#CNN=np.array(pd.read_excel(hpfile,sheet_name='majhon'))
#Hp=np.array(pd.read_excel(hpfile,sheet_name='comb_SM'))
buf=np.int64(np.array(pd.read_excel(hpfile,sheet_name='majhon'))[2:,2:])
Hp_truth,Hp_report=np.zeros(buf.shape[0]),buf[:,0]
for m in range(1,buf.shape[1]):
    Hp_truth,Hp_report=np.hstack([Hp_truth,np.ones(buf.shape[0])*m]),np.hstack([Hp_report,buf[:,m]])
    
Hp_pre=[]
Hp_ref=[]

for cl in range(buf.shape[1]):  
    for m in range(40):
        pp=np.zeros(buf.shape[1])
        tmp=Hp_report[Hp_truth==cl][classnum*m:classnum*(m+1)]
        for no in range(buf.shape[1]):pp[no]=np.sum(tmp==no)/classnum
        Hp_ref.append(cl)
        Hp_pre.append(pp)
    
Hp_ref=np.array(Hp_ref)
Hp_pre=np.array(Hp_pre)

    
mmc.plot_confusion_matrix(Hp_report,Hp_truth,sfname='HP',classes=mc.yls)
mmc.plot_roc(Hp_ref,Hp_pre,mc.yls,sfname='HP')

'''
mmc.data_boxplot(Y_test,final_y,mmc.yls,selc=-2,sfname='LSTM',dgns=[0,4])
mmc.data_boxplot(Y_test,final_y,mmc.yls,selc=-1,sfname='LSTM',dgns=[0,4])
mmc.plot_confusion_matrix(org_pred,orgLabel,sfname='ANN')
mmc.plot_accuracy(lossaccbf,sfname='ANN')
mmc.plot_roc(Y_test,final_y,seldrugs,sfname='ANN')   
#mmc.plot_concentration_boxplot(Y_test,final_y,seldrugs,selc=-1)   
mmc.plot_concentration_boxplot(Y_test,final_y,seldrugs,selc=-2,sfname='ANN')   
mmc.plot_R2(Y_test,final_y,seldrugs,am=80,selc=-1,sfname='ANN')
'''