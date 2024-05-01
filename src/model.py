import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from grakel.kernels import (WeisfeilerLehman, VertexHistogram, 
                            WeisfeilerLehmanOptimalAssignment, 
                            Propagation, GraphletSampling, 
                            RandomWalkLabeled, PyramidMatch)
from layers import *
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import time



H = lambda x: -torch.sum(x*x.log(),-1)
JSD = lambda x: H(x.mean(0))-H(x).mean(0)

class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        print(hparams)
        
        try: 
            in_features, hidden, num_classes, labels = hparams.in_features,  hparams.hidden,  hparams.num_classes, hparams.labels
        except:
            hparams = Namespace(**hparams)
            in_features, hidden, num_classes, labels = hparams.in_features,  hparams.hidden,  hparams.num_classes, hparams.labels

        #self.hparams=hparams
        self.save_hyperparameters(hparams)
       
#         assert(layers==1) #only works with 1 layer at the moment
        self.conv_layers = nn.ModuleList()
        self.vq_layers =nn.ModuleList()
    
        self.conv_layers.append(GKernel(hparams.nodes,labels,hidden,max_cc=self.hparams.max_cc,hops=hparams.hops, kernels=hparams.kernel, store_fit=True))
        
        n_kernels = len(hparams.kernel.split('+'))
        for i in range(1, hparams.layers):
            self.conv_layers.append(GKernel(hparams.nodes,hidden,hidden,max_cc=self.hparams.max_cc,hops=hparams.hops, kernels=hparams.kernel))
            
            self.vq_layers.append(SoftAss(hidden, hidden))
        
        activation = nn.LeakyReLU
        if hparams.activation == 'sigmoid':
          activation = nn.Sigmoid
          
#         print(hidden*n_kernels*hparams.layers)
#         print(type(hidden))
#         print(type(n_kernels))
#         print(type(hparams.layers))
#         print(type(num_classes))
        if hparams.mlp_layers==2:
            self.fc = nn.Sequential(nn.Linear(hidden*n_kernels*hparams.layers,hidden),activation(),nn.Linear(hidden,hidden),activation(),nn.Linear(hidden,num_classes))
        else:
            self.fc = nn.Sequential(nn.Linear(hidden*n_kernels,hidden),activation(),nn.Linear(hidden,num_classes))
        
        self.eye = torch.eye(hidden)
        self.lin = nn.Linear(hidden,hidden)
        
        self.automatic_optimization = False
        
        def _regularizers(x):
            jsdiv = hparams.jsd_weight*JSD(x.softmax(-1))
            return -jsdiv# + maxresp
        self.regularizers = _regularizers
            
        self.mask=nn.Parameter(torch.ones(hidden).float())
        
    def prefit(self,data):
        for l in self.conv_layers:
          if l.store_fit:
            with torch.no_grad():
              l.prefit(data.x,data.edge_index)
    
        
    def one_hot_embedding(self, labels):
        self.eye = self.eye.to(labels.device)
        return self.eye[labels] 

    def forward(self, data):
    
        if 'nidx' not in data.__dict__:
            data.nidx = None
            
        batch=data.batch
        edge_index=data.edge_index
        x=data.x
        
        egonets_data = [data.egonets_idx,data.egonets_edg]
        
        loss = x.sum().detach()*0
        
        responses = []
        for l,vq in zip(self.conv_layers,[None]+list(self.vq_layers)): #only works with one layer
#             if vq!=None:
#                 x = vq(x)
               
            x = l(x,edge_index,node_indexes=data.nidx,egonets_data=egonets_data,batch=batch,optimize=self.hparams.optimize_masks)
                
            if self.mask is not None:
                x = x*self.mask[None,:].repeat(1,x.shape[-1]//self.mask.shape[-1])
            
            responses.append(x)
        x = torch.cat(responses,-1)
#         x = responses[-1]
        
        pooling_op = None
        if self.hparams.pooling=='add': 
            pooling_op=global_add_pool
        if self.hparams.pooling=='max': 
            pooling_op=global_max_pool
        if self.hparams.pooling=='mean': 
            pooling_op=global_mean_pool
        
        return self.fc(pooling_op(x,batch)), responses, loss
    
    def configure_optimizers(self):
#         assert(len(self.conv_layers)==1)
        
        graph_params = set(self.conv_layers.parameters())
        cla_params = set(self.parameters())-graph_params
        optimizer = torch.optim.Adam([{'params': list(graph_params),'lr': self.hparams.lr_graph},\
                                      {'params': list(cla_params),'lr': self.hparams.lr}])
        
        return optimizer
    
    
    def training_step(self, train_batch, batch_idx):  
       
        data=train_batch
        
        optimizer = self.optimizers()
        
        optimizer.zero_grad()
            
        output, responses, vqloss = self(data)
        loss_ce = self.hparams.loss(torch.squeeze(output,-1), data.y)
        
        loss_jsd = torch.stack([self.regularizers(x) for x in responses]).mean()
#         loss_spa = self.hparams.sparsity * sum([sum([layer.Padd[fi].mean() for fi in range(self.hparams.hidden)]) -\
#                sum([layer.Prem[fi].mean() for fi in range(self.hparams.hidden)]) for layer in self.conv_layers])

        loss = loss_ce + loss_jsd #+ vqloss +  loss_spa
        loss.backward()
        optimizer.step()
        
        # for i,p in enumerate(self.conv_layers[0].parameters()):
            # if p.grad is None:
                # n=0
            # else:
                # n = torch.norm(p.grad)
                # self.log('p%d_%s' % (i,p.shape), n, on_step=False, on_epoch=True)
            
            
        acc = 100*torch.mean( (output.argmax(-1)==data.y).float()).detach().cpu()
        self.log('acc', acc, on_step=False, on_epoch=True)
        self.log('loss', loss.item(), on_step=False, on_epoch=True)
        self.log('loss_jsd', loss_jsd.item(), on_step=False, on_epoch=True)
        self.log('loss_ce', loss_ce.item(), on_step=False, on_epoch=True)
#         self.log('loss_spa', loss_spa.item(), on_step=False, on_epoch=True)


    # def training_epoch_end(self, training_step_outputs):      
        # self.logger.log_image(key="grads", images=[self.conv_layers[0].P[0].data.cpu(),(self.conv_layers[0].Padd[0]*self.conv_layers[0].temp).sigmoid().data.cpu()])
    
    
    def validation_step(self, train_batch, batch_idx):
        data=train_batch
        with torch.no_grad():
            output, x1, _ = self(data)
#             print(output.shape)
#             print(data.y.shape)
            loss = self.hparams.loss(torch.squeeze(output,-1), data.y)
            acc = 100*torch.mean( (output.argmax(-1)==data.y).float()).detach().cpu()
            self.log('val_loss', loss.item(), on_step=False, on_epoch=True)
            self.log('val_acc', acc, on_step=False, on_epoch=True)
 
    def test_step(self, train_batch, batch_idx):
        data=train_batch
        with torch.no_grad():
            output, x1, _ = self(data)
            loss = self.hparams.loss(torch.squeeze(output,-1), data.y)
            acc = 100*torch.mean( (output.argmax(-1)==data.y).float()).detach().cpu()
            self.log('test_loss', loss.item(), on_step=False, on_epoch=True)
#             self.log('test_loss', loss.item(), on_step=False, on_epoch=True)
            self.log('test_acc', acc, on_step=False, on_epoch=True)       
            
            
            
class SoftAss(nn.Module):
    def __init__(self, num_words, features_dim, softmax=True):
        super(SoftAss, self).__init__()
        
        if softmax:
            self.normalize = nn.Softmax(-1)
        else: 
            self.normalize = nn.Normalize(dim=-1)
            
        self.dict = nn.Parameter(torch.rand(num_words, features_dim).float(), requires_grad=True)
        self.codebook_init=False        
        
    def reset_codebook(self,x):
        if self.codebook_init:
            centroid, label = kmeans2(x.detach().cpu().numpy(), self.dict.detach().cpu().numpy(), minit='matrix')
        else:
            centroid, label = kmeans2( (x + torch.randn_like(x)*1e-4).detach().cpu().numpy(), self.dict.shape[0])
        self.dict.data = torch.from_numpy(centroid).float().to(x.device)
        self.codebook_init = True
        
        
    def forward(self, x):
        
        if self.training:
            self.reset_codebook(x)
            
#         _dict = self.normalize(self.dict)
        _x = self.normalize(x)

        res = torch.tensordot(_x, self.dict, dims=([1], [1]))

        return res