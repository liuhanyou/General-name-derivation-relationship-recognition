#构造数据集
import os
import paddle
from paddle.io import Dataset
import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings('ignore')

import pickle
#读取数据
def read_data(path):
    with open(path, "rb") as infile:
        data=pickle.load(infile)
    return data


#定义数据集
class myDataset(Dataset):
    def __init__(self):	
        # 数据集存放位置
        self.filepath = r"/home/aistudio/work/geo_derived_feature.pickle"  #dataset_dir为数据集实际路径，需要填写全路径
        self.data=read_data(self.filepath)
        self.p_topy={'Disjoint':0,'Within':1,'Intersect':2}
    def __getitem__(self, idx):
        source_name=self.data[idx]["source_name"]
        source_fclass=self.data[idx]["source_fclass"]
        derived_name=self.data[idx]["derived_name"]
        derived_fclass=self.data[idx]["derived_fclass"]
        topy_vec=self.data[idx]["topy_vec"]
        near_vec=self.data[idx]['near_vec']
        dif_start=self.data[idx]['dif_start']
        dif_end=self.data[idx]['dif_end']
        #获取地名拓扑特征向量
        vector=np.zeros(3,)
        morphological_topology=self.data[idx]['morphological_topology']
        index=self.p_topy[morphological_topology]
        vector[index]=1
        morphological_vector=vector.tolist()
        label=self.data[idx]['label']
        return {'source_name':source_name,'source_fclass':source_fclass,\
               'derived_name':derived_name,'derived_fclass':derived_fclass,\
                   'topy_vec':topy_vec,'near_vec':near_vec,\
                   'morphological_topology': morphological_vector,
                       'dif_start':dif_start,'dif_end':dif_end,'label':label}
    def __len__(self):
        return self.data.__len__()


from sklearn.model_selection import train_test_split
#划分训练、验证、测试数据集
input_dataset=myDataset()
print(len(input_dataset))
train_dataset,val_dataset=train_test_split(input_dataset,train_size=0.7,random_state=34,shuffle=True)
print('tran_dataset size',len(train_dataset),'val_dataset size',len(val_dataset))


import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import BertModel,ErnieModel
from paddlenlp.transformers import  BertTokenizer,ErnieTokenizer
#定义分词器
import paddle
import paddle.nn as nn

class MultiHeadAttention(nn.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, num_heads)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        # Linear transformations
        query = self.query_linear(inputs)
        #print('query',query.shape)
        key = self.key_linear(inputs)
        value = self.value_linear(inputs)
        # Reshape for multi-head attention
        query = query.reshape([batch_size, seq_len, self.num_heads, self.d_k])
        #print(query.shape)
        key = key.reshape([batch_size, seq_len, self.num_heads, self.d_k])
        value = value.reshape([batch_size, seq_len, self.num_heads, self.d_k])
        # Transpose dimensions for matrix multiplication
        query =paddle.transpose(query,perm=[0,2,1,3])
        key =paddle.transpose(key,perm=[0,2,1,3])
        value =paddle.transpose(value,perm=[0,2,1,3])
        # Scaled dot-product attention
        key =paddle.transpose(key,perm=[0,1,3,2])
        scores =paddle.matmul(query,key)  # [batch_size, num_heads, seq_len, seq_len]
        scores = scores / paddle.sqrt(paddle.to_tensor(self.d_k,dtype=paddle.float32))
        attention_weights =nn.Softmax()(scores)
        attention_output=paddle.matmul(attention_weights,value)  # [batch_size, num_heads, seq_len, d_k]
        # Concatenate and reshape
        attention_output=paddle.transpose(attention_output,perm=[0,1,3,2]).contiguous().reshape([batch_size, seq_len, self.d_model])
        # Linear transformation for final output
        #print(attention_output.shape)
        attention_output = self.output_linear(attention_output)
        #print(attention_output.shape)
        return attention_output

############
import paddle
import paddle.nn as nn

class MS_CAM(nn.Layer):
    def __init__(self, channels=18, r=2):
        super(MS_CAM, self).__init__()
        inter_channels =int(channels//r)
        self.conv1=nn.Conv1D(in_channels=channels,out_channels=inter_channels,kernel_size=1)
        self.batch_norm1=nn.BatchNorm(inter_channels)
        self.relu=nn.ReLU()
        self.conv2= nn.Conv1D(in_channels=inter_channels,out_channels=channels,kernel_size=1)
        self.batch_norm2=nn.BatchNorm(channels)
        self.AdaptiveAvgPool1D=nn.AdaptiveAvgPool1D(output_size=(1,))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x= paddle.transpose(x, perm=[0, 2, 1])
        x1=self.conv1(x)
        x1=self.batch_norm1(x1)
        x1=x1.squeeze(-1)
        x1=self.relu(x1)
        x1=x1.unsqueeze(-1)
        x1=self.conv2(x1)
        x1=self.batch_norm2(x1)
        #global
        x2=self.AdaptiveAvgPool1D(x)
        x2=self.conv1(x2)
        x2=self.batch_norm1(x2)
        x2=x2.squeeze(-1)
        x2=self.relu(x2)
        x2=x2.unsqueeze(-1)
        x2=self.conv2(x2)
        x2=self.batch_norm2(x2)
        x=x1+x2
        w=self.sigmoid(x)
        return  w*x
#########
import paddle
from paddle import nn
class SENet(nn.Layer):
    def __init__(self, in_channel, ratio=2):
        super(SENet, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool1D(in_channel)
        self.fc = nn.Sequential(
                 nn.Linear(in_channel,in_channel// ratio),
                 nn.ReLU(),
                 nn.Linear(in_channel//ratio,in_channel),
                 nn.Sigmoid())
    def forward(self, x):
        y=self.avg_pool(x)
        y=self.fc(y)
        return x*y


import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import BertModel,ErnieModel
from paddlenlp.transformers import  BertTokenizer,ErnieTokenizer
#定义分词器
tokenizer=BertTokenizer.from_pretrained('bert-base-cased')
#tokenizer=ErnieTokenizer.from_pretrained('ernie-2.0-base-en')
#定义网络模型
class Model1(nn.Layer):
      def __init__(self):
          super(Model1, self).__init__()
          self.model=BertModel.from_pretrained('bert-base-cased')
          #self.model=ErnieModel.from_pretrained('ernie-2.0-base-en')
          self.atten=MultiHeadAttention(21,21)
          self.MS_CAM=MS_CAM(21,2)
          self.SENet=SENet(21,2)
          self.conv1=nn.Conv1D(768,2,50)
          self.space_time_conv1=nn.Conv1D(1,1,6)
          self.layer_norm = paddle.nn.LayerNorm([21])
          self.dropout=paddle.nn.Dropout(p=0.2)
          self.linear=nn.Linear(21,2)
      def forward(self,input):
          #提取语义特征层
          inputs={
                'input_ids':input['input_ids'],
                'token_type_ids':input['token_type_ids'],
                'attention_mask':input['attention_mask']
                 }
          res=self.model(**inputs)
          logits=res[0]
          x= paddle.transpose(logits, perm=[0, 2, 1])
          x=self.conv1(x)
          x=F.max_pool1d(x,kernel_size=40)
          x= paddle.transpose(x, perm=[0, 2, 1])
          #提取时空特征层
          t=input['space_time']
          x=x.reshape([x.shape[0],-1])
          #简单特征融合
          m=paddle.concat([x,t],axis=-1)
          m=m.unsqueeze(1)
          #添加注意力机制
          #m=self.atten(m)
          #SENet
          m=self.SENet(m)
          #利用MS_CAM
          #m=self.MS_CAM(m)
          #AFF
          m=m.squeeze()
          x1=paddle.concat([x,paddle.zeros([t.shape[0],t.shape[1]])],axis=-1)
          t1=paddle.concat([paddle.zeros([x.shape[0],x.shape[1]]),t],axis=-1)
        #   print(x1.shape,t1.shape)
        #   print(m.shape)
          m=m*x1+(1-m)*t1
          #归一化
          #print(m.shape)
          m=self.layer_norm(m)
          m=self.dropout(m)
          #判别器
          #添加dropout层
          out=self.linear(m)
          return out   
model=Model1()




import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CrossEntropyCriterion(nn.Layer):
    def __init__(self, label_smooth_eps=None,label_num=0):
        """
        标签平滑的交叉熵损失
        输入：
            - label_smooth_eps: 标签平滑的参数
            - label_num: 标签类别数
        """
        super(CrossEntropyCriterion, self).__init__()
        self.label_smooth_eps = label_smooth_eps
        self.label_num= label_num

    def forward(self, predict, label):
        """
        前向计算
        输入：
            - predict: 模型的输出，维度为[批量大小, 标签类别数].
            - label: 标签，维度为[批量大小, 标签类别数]
        """
        # 标签平滑
        if self.label_smooth_eps:
            #print(F.one_hot(x=label, num_classes=self.label_num))
            label = F.label_smooth(label=F.one_hot(x=label, num_classes=self.label_num),\
                                   epsilon=self.label_smooth_eps)
            #print(12,label)
            # 经过标签平滑后，label的维度为[批量大小, 序列长度, 目标语言词典大小]
        # 交叉熵损失
        loss = F.cross_entropy(
            input=predict,
            label=label,
            reduction='none',
            soft_label=True if self.label_smooth_eps else False)
        # 返回 总损失，平均损失，非填充词元的数目
        return loss.mean()
# 平滑系数


from paddlenlp.data import Dict, Stack, Pad
from paddle.io import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from visualdl import LogWriter
import paddle.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import numpy as np

stack= Stack()
#定义模型训练器
class Trainer():
    def __init__(self,config):
        self.config=config
        self.input_maxlen=config['input_maxlen']  
        self.label_maxlen=config['label_maxlen']
        self.model_savepath=config['model_savepath']
        self.log_path=config['log_path']
        self.best_score=0
        self.start_epoch=0
        self.resume=config['resume']  
        self.prompt=config['prompt']
        self.model=config['model']
        self.label_smooth_eps=config['label_smooth_eps']
        self.label_num=config['label_num']
        self.tokenizer=config['tokenizer']
        self.optim=paddle.optimizer.AdamW(learning_rate=config['learning_rate'],\
                                         weight_decay=config['weight_decay'],\
                                         parameters=self.model.parameters(), epsilon=1e-6)
        #数据加载
        self.train_dataloader=DataLoader(config['train_dataset'], batch_size=config['batch_size'],shuffle=True,drop_last=True,collate_fn=self.Collate_fn,num_workers=1)
        self.val_dataloader=DataLoader(config['val_dataset'], batch_size=config['batch_size'],shuffle=True,drop_last=True,collate_fn=self.Collate_fn,num_workers=1)
        self.loss_fct=CrossEntropyCriterion(self.label_smooth_eps,self.label_num)
        #定义
        self.scheduler=paddle.optimizer.lr.LinearWarmup(learning_rate=config['learning_rate'], warmup_steps=len(self.train_dataloader), start_lr=0, end_lr=5e-5,verbose=True) 
    def Collate_fn(self,batch):     
         #对输入字符串进行分词编码
         input_ids_list=[]
         token_type_list=[]
         attention_mask_list=[]
         space_time_list=[]
         labels_list=[]
         for batch_dict in batch:
              source_name=batch_dict['source_name']
              source_fclass=batch_dict['source_fclass']
              derived_name=batch_dict['derived_name']
              derived_fclass=batch_dict['derived_fclass']
              topy_vec=batch_dict['topy_vec']
              near_vec=batch_dict['near_vec']
              dif_start=batch_dict['dif_start']
              dif_end=batch_dict['dif_end']
              morph_topo=batch_dict['morphological_topology']
              label=batch_dict['label']
              #提取语言特征
            #   #prompt0
            #   sent1="derived fclass:{} source fclass:{}.".format(derived_fclass,source_fclass)
            #   sent2="derived place name:{} source place name:{}".format(derived_name,source_name)
              #prompt1
              sent1="derived fclass:{} derived place name:{}.".format(derived_fclass,derived_name)
              sent2="source fclass:{} source place name:{}".format(source_fclass,source_name)
              input_dict=self.tokenizer(text=[sent1], \
                                      text_pair=[sent2],\
                                      max_length=100,\
                                    padding='max_length',\
                                       truncation=True,
                                       return_special_tokens=True,
                                       return_attention_mask=True,
                                      return_tensors='pd')
              #构建批次样本                       
              input_ids=paddle.to_tensor(input_dict['input_ids'])
              token_type_ids=paddle.to_tensor(input_dict['token_type_ids'])
              attention_mask=paddle.to_tensor(input_dict['attention_mask'])
              #构建语义空间特征
              #print(dif_start,dif_end)
              #space_time_feat=topy_vec+near_vec
              #添加地名拓扑特征
              space_time_feat=topy_vec+near_vec+morph_topo+dif_start+dif_end
              #语义时空关系
              #space_time_feat=topy_vec+near_vec+morph_topo+dif_start+dif_end
              #print(space_time_feat)
              space_time_feat=paddle.to_tensor(space_time_feat)
              #构建批次
              input_ids_list.append(input_ids[0])
              token_type_list.append(token_type_ids[0])
              attention_mask_list.append(attention_mask[0])
              space_time_list.append(space_time_feat)
              labels_list.append(paddle.to_tensor(label))
         #拼接成一整张输入张量
         input_ids_list=stack(input_ids_list)
         token_type_list=stack(token_type_list)
         attention_mask_list=stack(attention_mask_list)
         space_time_list=stack(space_time_list)
         label_list=stack(labels_list)
         return  {  'input_ids':input_ids_list,\
                    'attention_mask':attention_mask_list,\
                    'token_type_ids':token_type_list,\
                    'space_time':space_time_list,\
                    'labels':label_list}

    def load_model_optim_scheduler(self):
        print('模型已加载！')
        check_point_dict=paddle.load(os.path.join(self.model_savepath,'best_score.pkl'))
        print(check_point_dict)
        self.best_score=check_point_dict['best_score']
        self.start_epoch=check_point_dict['metric']['epoch']+1
        self.model.set_state_dict(paddle.load(os.path.join(self.model_savepath,'model.pdparams')))
        #self.optim.set_state_dict(paddle.load(os.path.join(self.model_savepath,'optim.pdopt')))
        
        return 
    def save_check_point(self,metric):
        # paddle.save(self.model.state_dict(),os.path.join(self.model_savepath,'model.pdparams'))
        self.model.train()
        print('模型已保存！')
        # 保存Layer参数
        paddle.save(self.model.state_dict(), os.path.join(self.model_savepath,"model.pdparams"))
        # 保存优化器参数
        paddle.save(self.optim.state_dict(), os.path.join(self.model_savepath,'optim.pdopt'))
        # 保存检查点checkpoint信息
        paddle.save({'metric':metric},os.path.join(self.model_savepath,'best_score.pkl'))
        return
    def compute_loss(self,input_dict):
        logits=self.model(input_dict)
        loss=self.loss_fct(logits,input_dict['labels'])
        #print(loss)
        return loss
        
    def compute_metrics(self,input_dict):
        logits=self.model(input_dict)
        preds=paddle.argmax(logits,axis=-1)
        label_ids=input_dict['labels'].reshape([-1])
        loss=self.loss_fct(logits,input_dict['labels'])
        #计算评价指标
        accuracy = accuracy_score(label_ids.cpu(), preds.cpu())
        precision = precision_score(label_ids.cpu(), preds.cpu(), average='weighted')
        recall = recall_score(label_ids.cpu(), preds.cpu(), average='weighted')
        f1 = f1_score(label_ids.cpu(), preds.cpu(), average='weighted') 
        #print(accuracy,precision,recall,f1)
        return  accuracy,precision,recall,f1,loss
    def model_train_eval(self):
        with LogWriter(logdir=self.log_path) as writer:
            for epoch in range(self.start_epoch,self.config['epoch']):
                #模型批次训练
                loss_batch=self.model_batch_train()
                # #更新学习率
                self.scheduler.step()
                print('epoch',epoch,"train/loss",loss_batch)
                writer.add_scalar(tag="train/loss", step=epoch, value=loss_batch)
                # # #模型批次评估
                acc,pre,rec,f1,loss=self.model_batch_eval()
                #print(acc,pre,rec,f1)
                writer.add_scalar(tag="acc", step=epoch, value=acc)
                writer.add_scalar(tag="pre", step=epoch, value=pre)
                writer.add_scalar(tag="rec", step=epoch, value=rec)
                writer.add_scalar(tag="f1", step=epoch, value=f1)
                writer.add_scalar(tag="val/loss", step=epoch,value=loss)
                if self.best_score<f1:
                    self.best_score=f1
                    metric={}
                    metric['accuracy']= acc
                    metric['precision']=pre
                    metric['recall']=rec
                    metric['f1']=f1
                    metric['epoch']=epoch
                    print('epoch',metric)
                    self.save_check_point(metric)
                    
        return 
    def model_batch_train(self):
        # 设置训练模式
        self.model.train()
        loss_batch=0
        for batch_id,input_dict in enumerate(self.train_dataloader):
            loss=self.compute_loss(input_dict)
            loss_batch+=loss.item()
            #梯度清零
            self.optim.clear_grad()
            #计算梯度，反向传播
            loss.backward()
            #更新参数
            self.optim.step()
        loss_batch/=(batch_id+1)
        return  loss_batch
    @paddle.no_grad()
    def model_batch_eval(self):
        #设置评估模式
        self.model.eval()
        loss_batch=0
        acc_eval,pre_eval,rec_eval,f1_eval=0,0,0,0
        for batch_id, input_dict in enumerate(self.val_dataloader):
             accuracy,precision,recall,f1,loss=self.compute_metrics(input_dict)
             loss_batch+=loss.item()
             acc_eval+=accuracy
             pre_eval+=precision
             rec_eval+=recall
             f1_eval+=f1
        return  acc_eval/(batch_id+1),pre_eval/(batch_id+1),rec_eval/(batch_id+1),f1_eval/(batch_id+1),loss_batch/(batch_id+1)



#模型评估
def evalate(data,model):
    #字符串编码
    preds=[]
    labels=[]
    res=[]
    p_topy={'Disjoint':0,'Within':1,'Intersect':2}
    for batch_dict in data:
        source_name=batch_dict['source_name']
        source_fclass=batch_dict['source_fclass']
        derived_name=batch_dict['derived_name']
        derived_fclass=batch_dict['derived_fclass']
        topy_vec=batch_dict['topy_vec']
        near_vec=batch_dict['near_vec']
        dif_start=batch_dict['dif_start']
        dif_end=batch_dict['dif_end']
        morph_topo=batch_dict['morphological_topology']
        vector=np.zeros(3,)
        index=p_topy[morph_topo]
        vector[index]=1
        morph_topo=vector.tolist()
        label=batch_dict['label']
        #提取语言特征
        #prompt0
        # sent1="derived fclass:{} source fclass:{}.".format(derived_fclass,source_fclass)
        # sent2="derived place name:{} source place name:{}".format(derived_name,source_name)
        #prompt1
        sent1="derived fclass:{} derived place name:{}.".format(derived_fclass,derived_name)
        sent2="source fclass:{} source plac name:{}".format(source_fclass,source_name)
        input_dict=tokenizer(text=[sent1], \
                                      text_pair=[sent2],\
                                      max_length=100,\
                                    padding='max_length',\
                                       truncation=True,
                                       return_special_tokens=True,
                                       return_attention_mask=True,
                                      return_tensors='pd')
        #print(input_dict)
        #print(morph_topo++dif_start+dif_end)
        space_time_feat=topy_vec+near_vec+morph_topo+dif_start+dif_end
        #print(space_time_feat)
        space_time_feat=paddle.to_tensor([space_time_feat])
        #print(space_time_feat)
        input_dict['space_time']=space_time_feat
        #
        logits=model(input_dict)
        pred=paddle.argmax(logits,axis=-1).numpy()[0]
        preds.append(pred)
        labels.append(label)
        #显示效果 
        res.append({'derived_name':derived_name,'derived_fclass':derived_fclass,'source_name':source_name,'source_fclass':source_fclass,'pred':pred,'label':label})
    #计算评价指标
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted') 
    print(accuracy, precision,recall,f1)
    return res
def load_model(model,model_path):
    #加载模型
    check_point_dict=paddle.load(os.path.join(model_path,"model.pdparams"))
    model.set_state_dict(check_point_dict)
    #模型加载完毕
    print('模型加载完毕！')
    return model

if __name__=='__main__':
    #模型训练配置
    test_name='Sematic_Space_Time_SSD_AFF'
    model_path=r'/home/aistudio/work/relation/{}'.format(test_name)
    config={
            'model':model,
            'tokenizer':tokenizer,
            'resume':False,
            'model_savepath':model_path,
            'log_path':r'./log/{}'.format(test_name),
            'train_dataset':train_dataset,
            "val_dataset":val_dataset,
            'label_smooth_eps':0.1,
            'label_num':2,
            'prompt':'',
            'input_maxlen':40,
            'label_maxlen':20,
            'epoch':20,
            'batch_size':128,
            'learning_rate':1e-5,
            'weight_decay':8e-1}
    #训练器初始化
    model_trainer=Trainer(config)
    model_trainer.model_train_eval()
    #模型预测
    path=r'/home/aistudio/work/test.pickle'
    #加载测试数据
    data=read_data(path)
    #加载模型
    model=load_model(model,model_path)
    res=evalate(data,model)


