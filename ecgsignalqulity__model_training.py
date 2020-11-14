
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

class  conv1d_inception_block(nn.Module):

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
    
    
    


    
class R_1Dcnn_RCNN_block(nn.Module):
    def __init__(self, in_ch, out_ch, t=2):
        super(R_1Dcnn_RCNN_block, self).__init__()
        
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
'''
class Attention_block_self(nn.Module):

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
'''

class model_1d(nn.Module):

    def __init__(self, img_ch=1, output_ch=1, t=1):
        super(model_1d, self).__init__()

        n1 =6
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Maxpool0 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.Maxpool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.RRCNN1 = R_1Dcnn_RCNN_block(img_ch, filters[3], t=t)
        self.RRCNN2 = R_1Dcnn_RCNN_block(filters[3], filters[2], t=t)
        self.RRCNN3 = R_1Dcnn_RCNN_block(filters[2], filters[2], t=t)
        self.Softmax = nn.LogSoftmax()
        self.fc1 = nn.Linear(168 ,3)
    def forward(self, x):
        x=self.Maxpool0(x)
        e1 = self.RRCNN1(x)
        e2 = self.Maxpool2(e1)        
        e2 = self.RRCNN2(e2)      
        e3 = self.Maxpool2(e2)
        e3 = self.RRCNN3(e3)
        e3 = self.Maxpool2(e3)
        e4=self.Maxpool2(e3)
        e7= e4.view(e4.size(0),e4.size(1)*e4.size(2))
        out = self.fc1(e7)
        out =self.Softmax(out)
        return out


def train(x,model,los_train, acc_train):
    optimizer = torch.optim.SGD(model.parameters(),lr=0.0050,  momentum=0.90, dampening=0, weight_decay=0.0001, nesterov=False)
    epoch=x
    j=x
    model.train() 
    running_loss = 0.0
    train_correct=0.0

    for i,data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()  # zero the gradient buffers
        output =  model(inputs)
        accracy =np.mean( (torch.argmax(output.cpu(),1)==torch.argmax(labels.cpu(),1)).numpy())
        loss = criterion(output, torch.argmax(labels, dim=1))
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        train_correct+=accracy

        if i % np.ceil(len(trainloader.dataset)/128) == np.ceil(len(trainloader.dataset)/128)-1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f  train_acc:%.4f'%
                  (epoch + 1, i + 1, running_loss /(len(trainloader.dataset)/128), train_correct/(len(trainloader.dataset)/128)))
            los_train[j]= running_loss /(len(trainloader.dataset)/128)
            acc_train[j]= train_correct/(len(trainloader.dataset)/128)                     
            running_loss = 0.0
            train_correct=0.0         
    return los_train, acc_train

def validation(j,model,los_test, acc_test):
     model.eval()
     test_loss = 0.0
     test_correct=0.0
     for inputs1, labels1 in testloader:
        inputs1, labels1 = Variable(inputs1.cuda()), Variable(labels1.cuda())
        output =  model(inputs1)
        accracy1 =(torch.argmax(output.cpu(),1)==torch.argmax(labels1.cpu(),1)).numpy().sum()
        loss1 = criterion(output, torch.argmax(labels1, dim=1))
        test_loss+=loss1.item()
        test_correct+=accracy1
     test_loss /= (len(testloader.dataset)/128)
     test_correct/=(len(testloader.dataset))
     los_test[j]= test_loss
     acc_test[j]= test_correct
     print('test_loss:%.4f, test_correct%.4f'%(test_loss,test_correct))

def test():
     model.eval()
     test_loss = 0.0
     test_correct=0.0
     for inputs1, labels1 in testloader:
        inputs1, labels1 = Variable(inputs1.cuda()), Variable(labels1.cuda())
        output =  model(inputs1)
        accracy1 =(torch.argmax(output.cpu(),1)==torch.argmax(labels1.cpu(),1)).numpy().sum()
        loss1 = criterion(output, torch.argmax(labels1, dim=1))
        test_loss+=loss1.item()
        test_correct+=accracy1
     test_loss /= (len(testloader.dataset)/128)
     test_correct/=(len(testloader.dataset))
     print('test_loss:%.4f, test_correct%.4f'%(test_loss,test_correct))     


data=h5py.File('ecgs.mat')
A=data['A']
B=data['B']
C=data['C']

ecga=np.vstack((A,B,C))
label1=np.zeros((3000,1))
label2=np.zeros((3000,1))+1
label3=np.zeros((3000,1))+2
labela=np.vstack((label1,label2,label3))
label1=[]
label2=[]
label3=[]
A=[]
B=[]
C=[]

file_number=0
sfolder = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
for traindata, testdata in sfolder.split(ecga,labela):
    print(testdata)
    file_number=file_number+1
    ecgc=ecga[traindata]
    ecgt=ecga[testdata]
    labelc=labela[traindata]
    labelt=labela[testdata]
    for FF in range(len(ecgc)):
        ecgc[FF,:]=butthigh(zscore(ecgc[FF,:]),400)
    for FF1 in range(len(ecgt)): 
        ecgt[FF1,:]=butthigh(zscore(ecgt[FF1,:]),400)
        
    ecgc=torch.FloatTensor(ecgc)
    ecgc=ecgc.unsqueeze(1)
    labelc=to_categorical(labelc)
    labelc=torch.FloatTensor(labelc)
    deal_dataset = TensorDataset(ecgc,labelc)
    trainloader=DataLoader(dataset=deal_dataset,batch_size=128,shuffle=True,num_workers=0)    
    
    ecgt=torch.FloatTensor(ecgt)
    ecgt=ecgt.unsqueeze(1)    
    labelt=to_categorical(labelt)
    labelt=torch.FloatTensor(labelt)
    deal_test_dataset = TensorDataset(ecgt,labelt)
    testloader=DataLoader(dataset=deal_test_dataset,batch_size=128,shuffle=True,num_workers=0) 

    model=model_1d()
    model = model.cuda()
    #torchsummary.summary(model.cuda(), input_size=(1,512,241))
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    los_train=np.zeros(200)
    acc_train=np.zeros(200)
    los_test=np.zeros(200)
    acc_test=np.zeros(200)
    for x in range(200):
        los_train, acc_train=train(x,model,los_train, acc_train)
        validation(x,model,los_test, acc_test)
        if acc_test[x]>0.85:
            matname='conmodel_'+str(file_number)+'_epoch'+str(x)+'.pkl'
            torch.save(model, matname)
    filename_save='c11inception_10foldaccloss_model'+str(file_number)+'.mat'
    sio.savemat(filename_save, {'los_train':los_train,'acc_train':acc_train,'los_test':los_test,'acc_test':acc_test})
        



