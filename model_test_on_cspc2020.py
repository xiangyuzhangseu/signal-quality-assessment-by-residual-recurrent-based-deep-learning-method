# -*- coding: utf-8 -*-


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import scipy.io as sio
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torch.autograd import Variable
import numpy as np
import h5py
from sklearn.preprocessing import minmax_scale 
import torch.optim as optim
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from scipy.fftpack import fft, ifft
import scipy.signal as signal
def zscore(data):
    data_mean=np.mean(data)
    data_std=np.std(data, axis=0)
    if data_std!=0:
        data=(data-data_mean)/data_std
    else:
        data=data-data_mean
    return data
def butthigh(ecg, fs):
    # wp = 1
    # ws = 0.5
    # Rp = 0.3
    # Rs = 2
    # [N1, Wn1] = signal.buttord(wp / (fs / 2), ws / (fs / 2), Rp, Rs)
    # [N1, Wn1] = signal.buttord(wp / (fs*1.0 / 2), ws / (fs*1.0 / 2), Rp, Rs)
    
    # [b1, a1] = signal.butter(N1, Wn1, 'high')
    b1 = np.array([0.995155038209359, -1.99031007641872, 0.995155038209359])
    a1 = np.array([1, -1.99028660262621, 0.990333550211225])
    ecg_copy = np.copy(ecg)
    ecg1 = signal.filtfilt(b1, a1, ecg_copy)
    return ecg1
def hobalka(ecg1, fs, fmin, fmax):
    ecg = np.copy(ecg1)
    n = len(ecg)
    ecg_fil = fft(ecg)
    if fmin > 0:
        imin = int(fmin / (fs / n))
    else:
        imin = 1
        ecg_fil[0] = ecg_fil[0] / 2
    if fmax < fs / 2:
        imax = int(fmax / float(fs / n))
    else:
        imax = int(n / 2)
    hamwindow = np.hamming(imax - imin)
    hamsize = len(hamwindow)
    yy = np.zeros(len(ecg_fil), dtype=complex)
    istred = int((imax + imin) / 2)
    dolni = np.arange(istred-1, imax)
    ld = len(dolni)
    yy[0: ld] = np.multiply(ecg_fil[dolni - 1], hamwindow[int(np.floor(hamsize / 2)) - 1: hamsize])
    horni = np.arange(imin-1, istred-1)
    lh = len(horni)
    end = len(yy)
    yy[end - lh - 1: end - 1] = np.multiply(ecg_fil[horni], hamwindow[0: int(np.floor(hamsize / 2))])
    ecg_fil = abs(ifft(yy)) * 2
    return ecg_fil
#数据准备及生成loader形式   
class  conv1d_inception_block(nn.Module):
    """
    Convolution Block 1d
    """
    def __init__(self, in_ch, out_ch):
        super(conv1d_inception_block, self).__init__()
        
        self.conv1_1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Dropout(0.3))
        self.conv1_3 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=True),
            nn.Dropout(0.3))
        self.conv1_5 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=True),
            nn.Dropout(0.3))
        self.conv= nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Dropout(0.3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU())
    def forward(self, x):

        x1 = self.conv1_1(x)
        x3 = self.conv1_3(x)
        x5 = self.conv1_5(x)
        return self.conv(x1+x3+x5)    
class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
      '''
            conv1d_inception_block(out_ch,out_ch),
            nn.Dropout(0.2),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Dropout(0.2),
            '''
    """
    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()
        #self.drop_layer = nn.Dropout(0.5)
        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
          
            conv1d_inception_block(out_ch,out_ch),
            nn.Dropout(0.3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True))
    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            out = self.conv1_1(x1 + x)
        return out
    
class Residual_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
      '''
            conv1d_inception_block(out_ch,out_ch),
            nn.Dropout(0.2),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Dropout(0.2),
            '''
    """
    def __init__(self, out_ch, t=2):
        super(Residual_block, self).__init__()
        #self.drop_layer = nn.Dropout(0.5)
        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
          
            conv1d_inception_block(out_ch,out_ch),
            nn.Dropout(0.3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True))
    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.conv(x1)
        out=self.conv1_1(x1+x)
        return out  
    
    
    


    
class R_inception_RCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """
    def __init__(self, in_ch, out_ch, t=2):
        super(R_inception_RCNN_block, self).__init__()
        
        self.RCNN1 = nn.Sequential(
            Recurrent_block(out_ch, t=t))
        
        self.RCNN2 = nn.Sequential(
            conv1d_inception_block(out_ch, out_ch))
        
        self.RCNN3 =nn.Sequential(
            Residual_block(out_ch, out_ch))

        
        self.Conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.Conv1_1 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU())
    def forward(self, x):
        x=self.Conv (x)
        x1 = self.RCNN3(x)
        x2 = self.RCNN2(x)
        x3 = self.RCNN1(x)
        out = self.Conv1_1(x3+x2+x1)
        return out
class Attention_block_self(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_l, F_int):
        super(Attention_block_self, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.Tanh = nn.Tanh()

    def forward(self,  x):
        x1 = self.W_g(x)
        psi = self.Tanh(x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class model_1d(nn.Module):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """
    def __init__(self, img_ch=1, output_ch=1, t=1):
        super(model_1d, self).__init__()

        n1 =6
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Maxpool0 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.Maxpool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.RRCNN1 = R_inception_RCNN_block(img_ch, filters[3], t=t)

        self.RRCNN2 = R_inception_RCNN_block(filters[3], filters[2], t=t)

        #self.RRCNN3 = R_inception_RCNN_block(filters[2], filters[2], t=t)
        self.RRCNN3 = R_inception_RCNN_block(filters[2], filters[2], t=t)
        #self.RRCNN4 = RRCNN_block(filters[3], filters[1], t=t)
        self.Attention_block=Attention_block_self( F_l=filters[2], F_int=2)
        self.Softmax = nn.LogSoftmax()
        self.fc1 = nn.Linear(168 ,3)
        #self.fc2 = nn.Linear(3 ,3)
        #self.att= LinearSelfAttn()
        #self.fc2 = nn.Linear(24,1)
    def forward(self, x):
        x=self.Maxpool0(x)
        e1 = self.RRCNN1(x)
        #e1 = self.drop_layer(e1)

        e2 = self.Maxpool2(e1) 
         
        e2 = self.RRCNN2(e2)
        #e2 = self.drop_layer(e2)
        
        e3 = self.Maxpool2(e2)
        
        
        e3 = self.RRCNN3(e3)
        e3 = self.Maxpool2(e3)
        '''
        #e3 = self.drop_layer(e3)
        
        e4 = self.Maxpool2(e3)
        e4 = self.RRCNN3(e4)
        
        e4= self.Maxpool2(e3)
        '''
        e4=self.Maxpool2(e3)

        
        #e4 = self.Attention_block(x=e4)
        
        
        e7= e4.view(e4.size(0),e4.size(1)*e4.size(2))
        #e7=e4.view(-1, e4.view(-1).size(0))
        out = self.fc1(e7)
        #out = self.fc2(out)
        out =self.Softmax(out)
        #out= out.view(-1,24)
        #out=self.fc2(out)
        return out


def test(model,testloader):
     model.eval()
     test_loss = 0.0
     test_correct=0.0
     labelpredict=[]
     label=[]
     data=[]
     for inputs1, labels1 in testloader:
        inputs1, labels1 = Variable(inputs1.cuda()), Variable(labels1.cuda())
        output =  model(inputs1)
        data.append(inputs1.cpu().numpy())
        label.append(torch.argmax(labels1.cpu(),1).numpy())
        labelpredict.append(torch.argmax(output.cpu(),1))
     test_correct/=(len(testloader.dataset))
     return labelpredict ,data


data=h5py.File('ecgpart_04.mat')
ecga=data['ecgpart']

#useless-------------------------------------------------------------
label1=np.zeros((len(ecga)-6000,1))
label2=np.zeros((3000,1))+1
label3=np.zeros((3000,1))+2
labelt=np.vstack((label1,label2,label3))
#---------------------------------------------------------------------


for FF1 in range(len(ecga)): 
    ecga[FF1,:]=butthigh(zscore(ecga[FF1,:]),400)
ecgt=torch.FloatTensor(ecga)
ecgt=ecgt.unsqueeze(1)    
labelt=to_categorical(labelt)
labelt=torch.FloatTensor(labelt)
deal_test_dataset = TensorDataset(ecgt,labelt)
testloader=DataLoader(dataset=deal_test_dataset,batch_size=128,shuffle=False,num_workers=0) 
modelname='trained_model_RRM.pkl'
model=torch.load(modelname)
model.eval()
labelpredict,testdata11=test(model,testloader)
j1=[]
for j in labelpredict:
    j2=j.numpy()
    j1.extend(j2)
matname_result='2020label_04_result.mat'
sio.savemat(matname_result,{"predict":j1})

