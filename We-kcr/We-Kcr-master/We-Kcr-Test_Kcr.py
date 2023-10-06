#!/usr/bin/env python3

#!/usr/bin/env python3

#!/usr/bin/env python3

#!/usr/bin/env python3

#!/usr/bin/env python3
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import sys
import torch.nn.functional as F
import os
import random
import re
import torch
import torch.nn as nn ##########
import torch.utils.data as Data
from torch.autograd import Variable
from torch import optim
from sklearn.model_selection import train_test_split
import itertools

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix,recall_score,matthews_corrcoef,roc_curve,roc_auc_score,auc,precision_recall_curve
# from torch.utils.tensorboard import SummaryWriter
import seaborn as sns

#----->>
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle

#-------------------------------------->>>>

def load_data_bicoding(Path):
	data = np.loadtxt(Path,dtype=list)
	data_result = []
	for seq in data:
		seq = str(seq.strip('\n'))
		data_result.append(seq)
	return data_result 
	
	
def transform_token2index(sequences): 
	token2index = pickle.load(open('./data/residue2idx2.pkl', 'rb'))
	print(token2index)
	
	for i, seq in enumerate(sequences):
		sequences[i] = list(seq)
	
	token_list = list()
	max_len = 0
	for seq in sequences:
		seq_id = [token2index[residue] for residue in seq]
		token_list.append(seq_id)
		if len(seq) > max_len:
			max_len = len(seq)
	
	print('-' * 20, '[transform_token2index]: check sequences_residue and token_list head', '-' * 20)
	print('sequences_residue', sequences[0:5]) 
	print('token_list', token_list[0:5])     
	return token_list, max_len

def make_data_with_unified_length(token_list,max_len):
	token2index = pickle.load(open('./data/residue2idx2.pkl', 'rb'))
	data = []
	for i in range(len(token_list)):
		token_list[i] = [token2index['[CLS]']] + token_list[i] + [token2index['[SEP]']] 
		n_pad = max_len - len(token_list[i])
		token_list[i].extend([0] * n_pad) 
		data.append(token_list[i])
	
	print('-' * 20, '[make_data_with_unified_length]: check token_list head', '-' * 20)
	print('max_len + 2', max_len)
	print('token_list + [pad]', token_list[0:5])
	
	return data


	
def load_train_val_bicoding(path_pos_data, path_neg_data):
	
	sequences_pos = load_data_bicoding(path_pos_data)
	sequences_neg = load_data_bicoding(path_neg_data)
	
	token_list_pos, max_len_pos = transform_token2index(sequences_pos) #打印1次
	token_list_neg, max_len_neg = transform_token2index(sequences_neg) #打印1次
	# token_list_train: [[1, 5, 8], [2, 7, 9], ...]
	max_len = max(max_len_pos,max_len_neg)
	
	Positive_X = make_data_with_unified_length(token_list_pos,max_len)
	Negitive_X = make_data_with_unified_length(token_list_neg,max_len)
	
	data_train = np.array([_ + [1] for _ in Positive_X] + [_ + [0] for _ in Negitive_X])
	
	np.random.seed(42)
	np.random.shuffle(data_train)

	X = np.array([_[:-1] for _ in data_train])
	y = np.array([_[-1] for _ in data_train])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0/10, random_state=42)

	return X_train, y_train, X_test, y_test


def encoding(samples):
    try:
        with open('./AAindex/AAindex_normalized.txt') as f:
            records = f.readlines()[1:]
        # AA index 氨基酸理化属性
        AA_aaindex = 'ARNDCQEGHILKMFPSTWYV'   # 不同残基顺序
        AAindex = []
        AAindexName = []
        for i in records:
            AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
            AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)
        # 对问题最有影响的29种aaindex，变成了list
        props = 'FINA910104:LEVM760101:JACR890101:ZIMJ680104:RADA880108:JANJ780101:CHOC760102:NADH010102:KYTJ820101:' \
                'NAKH900110:GUYH850101:EISD860102:HUTJ700103:OLSK800101:JURD980101:FAUJ830101:OOBM770101:GARJ730101:' \
                'ROSM880102:RICJ880113:KIDA850101:KLEP840101:FASG760103:WILM950103:WOLS870103:COWR900101:KRIW790101:' \
                'AURR980116:NAKH920108'.split(':')
        # 生成大表
        if props:
            tmpIndexNames = []
            tmpIndex = []
            for p in props:
                if AAindexName.index(p) != -1:
                    tmpIndexNames.append(p)
                    tmpIndex.append(AAindex[AAindexName.index(p)])
            if len(tmpIndexNames) != 0:
                AAindexName = tmpIndexNames
                AAindex = tmpIndex

        index = {}
        for i in range(len(AA_aaindex)):
            index[AA_aaindex[i]] = i

        """
        从BLOCKS数据库中选取具有良好保守型的蛋白家族，从中统计氨基酸发生替换的相对频率和概率所构成的矩阵。
        BLOSUM-n中，n越小，表示氨基酸相似的可能性越小。相似序列比较选用n值大的矩阵，进化距离远的序列比较选用n值小的矩阵。
        BLOSUM-62用来比较62％相似度的序列，BLOSUM-80用来比较80％左右相似度的序列。
        """
        blosum62 = {
            'A': [4,  -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
            'R': [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
            'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
            'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
            'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
            'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
            'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
            'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
            'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
            'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],  # I
            'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],  # L
            'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
            'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],  # M
            'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1], # F
            'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
            'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
            'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
            'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3], # W
            'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1], # Y
            'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
            '-': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # -
        }

        # 归一化
        for key in blosum62:
            for j, value in enumerate(blosum62[key]):
                blosum62[key][j] = round((value+4)/15, 3)
                # (-1,1)区间

        encoding_aaindex = []
        for i in samples:
            name, sequence, label = i[0], i[1], i[2]
            code = [name, label]
            for aa in sequence:
                if aa == '-':
                    for j in AAindex:
                        code.append(0)
                    continue
                for j in AAindex:
                    code.append(j[index[aa]])
            encoding_aaindex.append(code)
        # print('#' * 10, '\n', "encoding_AAindex_____", encoding_aaindex)

        AA = 'ARNDCQEGHILKMFPSTWYV'
        encoding_binary = []
        for i in samples:
            name, sequence, label = i[0], i[1], i[2]
            code = [name, label]
            for aa in sequence:
                if aa == '-':
                    code = code + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    code += [0.05] * 9
                    continue
                for aa1 in AA:
                    tag = 1 if aa == aa1 else 0
                    code.append(tag)
                # 补齐
                code += [0.05] * 9
            encoding_binary.append(code)

        # 查表
        encoding_blosum = []
        for i in samples:
            name, sequence, label = i[0], i[1], i[2]
            code = [name, label]
            for aa in sequence:
                code = code + blosum62[aa]
                # 补齐，blosum原本只有20维，补上9维
                code += [0.267] * 9
            encoding_blosum.append(code)

        encoding_aaindex = np.array(encoding_aaindex)
        encoding_binary = np.array(encoding_binary)
        encoding_blosum = np.array(encoding_blosum)

        encodings = np.hstack((encoding_aaindex[:, 1:], encoding_binary[:, 2:], encoding_blosum[:, 2:]))
        #encodings = encoding_binary[:, 1:]

        return True, encodings.astype(np.float32)
    except Exception as e:
        return False, None

def load_test_bicoding(path_pos_data, path_neg_data, n_pos, n_neg):
	sequences_pos = load_data_bicoding(path_pos_data)
	sequences_neg = load_data_bicoding(path_neg_data)

	token_list_pos, max_len_pos = transform_token2index(sequences_pos)  # 打印1次
	token_list_neg, max_len_neg = transform_token2index(sequences_neg)  # 打印1次
	# token_list_train: [[1, 5, 8], [2, 7, 9], ...]
	max_len = max(max_len_pos, max_len_neg)

	Positive_X = make_data_with_unified_length(token_list_pos, max_len)
	Negitive_X = make_data_with_unified_length(token_list_neg, max_len)
	ok_encoding, Positive_X1 = encoding(n_pos)
	ok_encoding, Negitive_X1 = encoding(n_neg)

	data_train = np.array([_ + [1] for _ in Positive_X] + [_ + [0] for _ in Negitive_X])
	data_train1 = np.append(Positive_X1, Negitive_X1, axis=0)
	data_sum = np.append(data_train, data_train1, axis=1)
	np.random.seed(42)
	np.random.shuffle(data_sum)

	X_test = np.array([_[:31] for _ in data_sum])
	y_test = np.array([_[31] for _ in data_sum])
	X_test1 = np.array([_[32:] for _ in data_sum])
	return X_test, X_test1, y_test


def load_in_torch_fmt(X_train, y_train):

	X_train = torch.from_numpy(X_train).long()
	y_train = torch.from_numpy(y_train).long()

	X_test, y_test = shuffleData(X_train, y_train)
	return X_train, y_train



def save_checkpoint(state,is_best,OutputDir,test_index):
	if is_best:
		print('=> Saving a new best from epoch %d"' % state['epoch'])
		torch.save(state, OutputDir + '/' + str(test_index) +'_checkpoint1.pth.tar')
		
	else:
		print("=> Validation Performance did not improve")
		
def ytest_ypred_to_file(y, y_pred, out_fn):
	with open(out_fn,'w') as f:
		for i in range(len(y)):
			f.write(str(y[i])+'\t'+str(y_pred[i][0])+'\n')
			
			
def chunkIt(seq, num): 
	avg = len(seq) / float(num)
	out = []
	last = 0.0
	
	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg
		
	return out

def shuffleData(X, y):
	index = [i for i in range(len(X))]
	# np.random.seed(42)
	random.shuffle(index)
	new_X = X[index]
	new_y = y[index]
	return new_X, new_y

def round_pred(pred):
	list_result = []
	for i in pred:
		if i >0.5:
			list_result.append(1)
		elif i <=0.5:
			list_result.append(0)
	return torch.tensor(list_result)
	
def get_loss(logits, label, criterion):
	loss = criterion(fx.squeeze(), train_y.type(torch.FloatTensor).to(device))
	loss = (loss.float()).mean()
	# flooding method
	loss = (loss - 0.06).abs() + 0.06

	# multi-sense loss
	# alpha = -0.1
	# loss_dist = alpha * cal_loss_dist_by_cosine(model)
	# loss += loss_dist

	return loss
#--------------------------------------------------->>>
class BahdanauAttention(nn.Module):
	"""
	input: from RNN module h_1, ... , h_n (batch_size, seq_len, units*num_directions),
									h_n: (num_directions, batch_size, units)
	return: (batch_size, num_task, units)
	"""
	def __init__(self,in_features, hidden_units,num_task):
		super(BahdanauAttention,self).__init__()
		self.W1 = nn.Linear(in_features=in_features,out_features=hidden_units)
		self.W2 = nn.Linear(in_features=in_features,out_features=hidden_units)
		self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

	def forward(self, hidden_states, values):
		hidden_with_time_axis = torch.unsqueeze(hidden_states,dim=1)

		score  = self.V(nn.Tanh()(self.W1(values)+self.W2(hidden_with_time_axis)))
		attention_weights = nn.Softmax(dim=1)(score)
		values = torch.transpose(values,1,2)   # transpose to make it suitable for matrix multiplication

		context_vector = torch.matmul(values,attention_weights)
		context_vector = torch.transpose(context_vector,1,2)
		return context_vector, attention_weights
		
class Embedding(nn.Module):
	def __init__(self, vocab_size,d_model,max_len):
		super(Embedding, self).__init__()
		self.tok_embed = nn.Embedding(vocab_size, d_model)
		self.pos_embed = nn.Embedding(max_len, d_model)
		self.norm = nn.LayerNorm(d_model)

	def forward(self, x):
		seq_len = x.size(1) 
		pos = torch.arange(seq_len, device=device, dtype=torch.long) 
		pos = pos.unsqueeze(0).expand_as(x)

		embedding = self.pos_embed(pos)
		embedding = embedding + self.tok_embed(x) 
		embedding = self.norm(embedding)
		return embedding


class Adapt_emb_CNNLSTM_ATT(nn.Module):
	def __init__(self):
		super(Adapt_emb_CNNLSTM_ATT, self).__init__()
		kernel_size = 10
		max_len = 31
		d_model = 30
		vocab_size = 29

		self.embedding = Embedding(vocab_size, d_model, max_len)
		# ---------------->>>
		self.conv1 = nn.Sequential(
			nn.Conv1d(
				in_channels=30,  # input height
				out_channels=256,  # n_filters
				kernel_size=kernel_size),
			# dilation =2),     #!!!
			# padding = int(kernel_size/2)),
			# padding=(kernel_size-1)/2
			nn.ReLU(),  # activation
			# nn.MaxPool1d(kernel_size=2),
			nn.BatchNorm1d(256),
			nn.Dropout())

		self.lstm = torch.nn.LSTM(256, 128, 1, batch_first=True, bidirectional=True)  #
		self.Attention = BahdanauAttention(in_features=256, hidden_units=10, num_task=1)

		self.fc_task = nn.Sequential(
			nn.Linear(256, 128),
			nn.Dropout(0.5),
			nn.ReLU(),
			nn.Linear(128, 64),
		)
		self.classifier = nn.Linear(64, 1)

		# ---------------->>>
		out_channels = 128
		conv_kernel_size = 5
		pool_kernel_size = 2
		self.conv11 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=conv_kernel_size, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
		# self.dropout1 = nn.Dropout(0.5)
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=conv_kernel_size, stride=1,
					  padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=pool_kernel_size)
		)
		# self.dropout2 = nn.Dropout(0.5)
		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=conv_kernel_size, stride=1,
					  padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=pool_kernel_size)
		)
		# self.dropout3 = nn.Dropout(0.5)
		self.fc1 = nn.Linear(9 * out_channels, 64)
		self.fc2 = nn.Linear(64, 1)

	# ---------------->>>
	def forward(self, x, X):
		x = self.embedding(x)
		x = x.transpose(1, 2)
		batch_size, features, seq_len = x.size()

		x = self.conv1(x)
		x = x.transpose(1, 2)
		# rnn layer
		out1, (h_n, c_n) = self.lstm(x)
		h_n = h_n.view(batch_size, out1.size()[-1])
		context_vector, attention_weights = self.Attention(h_n, out1)
		reduction_feature = self.fc_task(torch.mean(context_vector, 1))
		representation = reduction_feature
		# logits_clsf = self.classifier(representation)
		# logits_clsf1 = logits_clsf
		# logits_clsf = torch.sigmoid(logits_clsf)
		# ---------------
		out = X.view(-1, 3, 29, 29)
		out = self.conv11(out)
		out = self.conv2(out)
		out = self.conv3(out)

		out = out.view(out.size(0), -1)

		out = F.relu(self.fc1(out))
		out = (out + representation)/2
		out = torch.sigmoid(self.fc2(out))
		return out


def calculateScore(y, pred_y):

	y = y.data.cpu().numpy()
	tempLabel = np.zeros(shape = y.shape, dtype=np.int32)
	
	for i in range(len(y)):
		if pred_y[i] < 0.5:
			tempLabel[i] = 0;
		else:
			tempLabel[i] = 1;
			
	accuracy = metrics.accuracy_score(y, tempLabel)
	
	confusion = confusion_matrix(y, tempLabel)
	TN, FP, FN, TP = confusion.ravel()
	
	sensitivity = recall_score(y, tempLabel)
	specificity = TN / float(TN+FP)
	MCC = matthews_corrcoef(y, tempLabel)
	
	F1Score = (2 * TP) / float(2 * TP + FP + FN)
#	precision = TP / float(TP + FP)
	
	precision = metrics.precision_score(y, tempLabel, pos_label=1)  
	recall = metrics.recall_score(y, tempLabel, pos_label=1)
	
	pred_y = pred_y.reshape((-1, ))
	
	# ROCArea = roc_auc_score(y, pred_y)
	ROCArea = metrics.roc_auc_score(y, pred_y)
	fpr, tpr, thresholds = roc_curve(y, pred_y)
	
	pre, rec, threshlds = precision_recall_curve(y, pred_y)
	pre = np.fliplr([pre])[0] 
	rec = np.fliplr([rec])[0]  
	AUC_prec_rec = np.trapz(rec,pre)
	AUC_prec_rec = abs(AUC_prec_rec)
	
	return {'sn' : sensitivity, 'sp' : specificity, 'acc' : accuracy, 'MCC' : MCC, 'AUC' : ROCArea,'precision' : precision, 'F1' : F1Score, 'fpr' : fpr, 'tpr' : tpr, 'thresholds' : thresholds,'pre_recall_curve':AUC_prec_rec,'prec':pre,'reca':rec}

def Get_data(file):
    with open(file) as f:
        record = f.read()
    records = record.split('>')[1:]
    dataSet = []
    for fasta in records:
        array = fasta.split('\n')
        name, label, attribute = array[0].split('|', 2)[0], int(array[0].split('|', 2)[1]), array[0].split('|', 2)[2]
        myStr = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '-', array[1])
        dataSet.append([name, myStr, label, attribute])
    return dataSet
class DealDataset(Dataset):
    def __init__(self, np_data,label,np_data1):
        self.__np_data = np_data
        self.X1= np_data1
        self.X = torch.from_numpy(self.__np_data[:, 1:])
        self.y= label
        self.len = self.__np_data.shape[0]

    def __getitem__(self, index):
        return self.X1[index], self.X[index],self.y[index]

    def __len__(self):
        return self.len

if __name__ == '__main__':
	# Hyper Parameters------------------>>

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device_cpu = torch.device("cpu") 
	# wordvec_len =4
	
	k_folds = 10
	EPOCH = 50
	BATCH_SIZE = 128
	LR = 0.001
	kernel_size = 10
	
	#!!! Model

	net = Adapt_emb_CNNLSTM_ATT().to(device)


	print(net)  # net architecture
	
	train_loss_sum, valid_loss_sum, test_loss_sum= 0, 0, 0
	train_acc_sum , valid_acc_sum , test_acc_sum = 0, 0, 0
	
	test_acc = []  
	test_auc = []
	test_losses = []

	all_index = [0]
	species = 'Kcr'
	All_testing_result = []
	# for index_fold in all_index: #index_fold #test_index
		# species = 'IPs'
		# sits = 'ST'
		# print('-------------------->fold_%s'% index_fold)
	print('*'*45,'strat test ....' ,'折','*'*45)
	
	#保存结果的列表
	testing_result = []
	
	import os
	#Read_model_Dir = './Kcr_Model'
	Read_model_Dir = './Result/Adapt_Train_Kcr/fold_1'
	OutputDir = Read_model_Dir
	
	#读取
	test_pos_fa = './data/Kcr/bigtest-1.fa'
	test_neg_fa = './data/Kcr/bigtest-0.fa'
	n_pos = Get_data("./data/Kcr/bigtest-11.fa")
	n_neg = Get_data("./data/Kcr/bigtest-00.fa")
	

	X_test, X_test1,y_test = load_test_bicoding(test_pos_fa,test_neg_fa,n_pos,n_neg)
	X_test, y_test = load_in_torch_fmt(X_test, y_test)
	
	#------------------->>>
	print('Test 数据：',X_test.shape[0])
	torch_test, torch_test_y   = torch.tensor([]),torch.tensor([]) #用来保存所有数据
	#------------------->>>

	datasetloader = DealDataset(X_test1.astype(np.float32),y_test,X_test)
	test_loader = DataLoader(datasetloader, 128, shuffle=False)
	model = net
	
	
	# criterion = nn.CrossEntropyLoss()												
	criterion = nn.BCELoss(size_average=False)											

	
	#!!! Test
	checkpoint = torch.load(Read_model_Dir + '/' + '0_checkpoint1.pth.tar',map_location=torch.device('cpu'))
	print('---------load_model: checkpoint.pth.tar----->>>>')
	model = net#重建模型结构
	
	print('model loaded ---->>')
	model.load_state_dict(checkpoint['state_dict'])
	
	model.to(device)
	model.eval()
	test_loss = 0
	test_correct = 0 
	for step, (test_x, test_x1,test_y) in enumerate(test_loader):
		test_x = test_x.to(device)  # ----cuda
		test_x1 = test_x1.to(device)
		test_y = test_y.to(device)
		
		# optimizer.zero_grad()
		y_hat_test = model(test_x,test_x1)
		#loss = criterion(y_hat, test_y.to(device)).item()      # batch average loss 		       #C
		loss = criterion(y_hat_test.squeeze(), test_y.type(torch.FloatTensor).to(device)).item()   #B
		test_loss += loss * len(test_y)             # sum up batch loss 
		
		# pred = y_hat_test.max(1, keepdim = True)[1]                                       #C     
		pred_test = round_pred(y_hat_test.data.cpu().numpy()).to(device) 					#B
		# get the index of the max log-probability
		test_correct += pred_test.eq(test_y.view_as(pred_test)).sum().item()
		
		# pred_prob = y_hat.max(1, keepdim = True)[0]
		#pred_prob = y_hat[:,1] #																#C
		pred_prob_test = y_hat_test																#B
		torch_test = torch.cat([torch_test,pred_prob_test.data.cpu()],dim=0)
		torch_test_y = torch.cat([torch_test_y,test_y.data.cpu()],dim=0)
		
	test_losses.append(test_loss/len(X_test)) # all loss / all sample
	test_accuracy = 100.*test_correct/len(X_test)
	print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
		test_loss/len(X_test), test_correct, len(X_test), test_accuracy))

	test_acc.append(test_accuracy)


	test_loss_sum += test_losses[-1]
	test_acc_sum += test_acc[-1]

	testing_result.append(calculateScore(torch_test_y,torch_test.numpy()))
	All_testing_result.append(calculateScore(torch_test_y,torch_test.numpy()))
	
	out_test_file = OutputDir+'/test_result_Test.txt'
	ytest_ypred_to_file(torch_test_y.numpy(), torch_test.numpy(), out_test_file)
	
	
	auroc = metrics.roc_auc_score(torch_test_y.numpy(), torch_test.numpy())
	test_auc.append(auroc)

	
	print('average test loss:{:.4f}, average test accuracy:{:.3f}%'.format(test_loss_sum/len(all_index), test_acc_sum/len(all_index)))
	print('all test acc:{}'.format(test_acc))
	print('all test auc:{}'.format(test_auc))
	print('average test acc: {} #--->'.format(np.sum(test_acc)/len(all_index)))
	print('average test auc: {} #--->'.format(np.sum(test_auc)/len(all_index)))
	for a in All_testing_result:
		print(a)

		
		
		
		
		
		
		
	
	
	
	
