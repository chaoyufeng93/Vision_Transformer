import torch
import os
import math
import numpy as np
import random
import torch.nn.functional as F


# e.g., in CIFAR 10 img_size = 32, patch_num = how many patches you want

class Patch(torch.nn.Module):
  def __init__(self, img_size, patch_num, emb_dim):
    super(Patch, self).__init__()
    self.emb_dim = emb_dim
    self.patch_it = torch.nn.Conv2d(in_channels = 3, out_channels = emb_dim, kernel_size = img_size//patch_num, stride = img_size//patch_num)

  def forward(self, x):
    patch = self.patch_it(x).permute(0,2,3,1)
    patch = patch.contiguous().view(patch.shape[0],-1,self.emb_dim)
    return patch
    
    
class Positional_Emb(torch.nn.Module):
  def __init__(self, img_size, patch_num, emb_dim, dropout = 0):
    super(Positional_Emb, self).__init__()
    self.pos_emb = torch.nn.Parameter(torch.randn(1 + img_size*img_size//((img_size//patch_num)*(img_size//patch_num)), emb_dim))
    self.dropout = torch.nn.Dropout(p = dropout)
    self.reset_weigths()

  def reset_weigths(self):
    """reset weights"""
    for weight in self.parameters():
      torch.nn.init.xavier_normal_(weight)

  def forward(self, x):
    out = self.pos_emb.repeat(x.shape[0],1,1)
    out[:,1:,:] += x
    out = self.dropout(out)
    return out
    
    
class Attention(torch.nn.Module):
  def __init__(self, emb_dim, head, dropout = 0):
    super(Attention, self).__init__()
    self.emb_dim = emb_dim
    self.head = head
    self.softmax = torch.nn.Softmax(dim = -1)
    self.dropout = torch.nn.Dropout(p = dropout)

  #sent k.T in (transpose k before sent in forward)
  def forward(self, q, k, v):
    qk = torch.matmul(q, k) / math.sqrt(self.emb_dim//self.head)
    att_w = self.dropout(self.softmax(qk))
    out = torch.matmul(att_w, v)
    return out
    
    
class Multi_Head_ATT(torch.nn.Module):
    def __init__(self, emb_dim, multi_head = 1, mask = False, post_LN = True, dropout = 0):
      super(Multi_Head_ATT,self).__init__()
      self.head = multi_head
      self.emb_dim = emb_dim
      self.q_att = torch.nn.Linear(emb_dim, emb_dim, bias = False) 
      self.k_att = torch.nn.Linear(emb_dim, emb_dim, bias = False) 
      self.v_att = torch.nn.Linear(emb_dim, emb_dim, bias = False) 
      self.attention = Attention(emb_dim, multi_head, dropout = dropout)
      self.LN = torch.nn.LayerNorm(emb_dim, eps = 1e-6)
      self.WO = torch.nn.Linear(emb_dim, emb_dim)
      self.dropout = torch.nn.Dropout(p = dropout)

    def forward(self, q,k,v):
      res = q
      seq_len = q.shape[1]
      q, k, v = self.LN(q), self.LN(k), self.LN(v)
      if self.head == 1:
        q, k, v = self.q_att(q), self.k_att(k).permute(0,2,1), self.v_att(v)
        out = self.attention(q, k, v)
      else:
        # (b_s, seq_len, head, emb//head) > (b_s, head, seq_len, emb_dim//head)
        q = self.q_att(q).view(-1,seq_len,self.head,self.emb_dim//self.head).permute(0,2,1,3)
        k = self.k_att(k).view(-1,seq_len,self.head,self.emb_dim//self.head).permute(0,2,1,3).permute(0,1,3,2)
        v = self.v_att(v).view(-1,seq_len,self.head,self.emb_dim//self.head).permute(0,2,1,3)
        out = self.attention(q, k, v).permute(0,2,1,3).contiguous().view(-1,seq_len,self.emb_dim)
        out = self.WO(out)   
      out = self.dropout(out)
      out = out + res
      return out
      
 
class Feed_Forward(torch.nn.Module): 
  def __init__(self, emb_dim, dim_expan = 4, post_LN = True, dropout = 0):
    super(Feed_Forward,self).__init__()
    self.w1 = torch.nn.Linear(emb_dim, dim_expan*emb_dim)
    self.w2 = torch.nn.Linear(dim_expan*emb_dim, emb_dim)
    self.gelu = torch.nn.GELU()
    self.LN = torch.nn.LayerNorm(emb_dim, eps = 1e-6)
    self.dropout = torch.nn.Dropout(p = dropout)
  def forward(self,x):
    res = x
    x = self.LN(x)
    out = self.gelu(self.w1(x))
    out = self.w2(out)
    out = self.dropout(out)
    out = out + res
    return out
    
    
class Encoder(torch.nn.Module):
  def __init__(self, num_layer, emb_dim, head, dim_expan = 4, dropout = 0):
    super(Encoder, self).__init__()
    self.num_layer = num_layer
    self.attention = Multi_Head_ATT(emb_dim, multi_head = head, dropout = dropout)
    self.FF = Feed_Forward(emb_dim, dim_expan = dim_expan, dropout = dropout)
    self.connect1 = torch.nn.ModuleList([
                                         Multi_Head_ATT(emb_dim, multi_head = head, dropout = dropout) for i in range(num_layer - 1)
                                         ])
    self.connect2 = torch.nn.ModuleList([
                                         Feed_Forward(emb_dim, dim_expan = dim_expan, dropout = dropout) for i in range(num_layer - 1)
                                         ])

  def forward(self, x):
    out = self.FF(self.attention(x, x, x))
    for idx in range(self.num_layer - 1):
      out = self.connect1[idx](out, out, out)
      out = self.connect2[idx](out)
    return out
    
    
class ViTransformer(torch.nn.Module):
  def __init__(self, img_size, patch_num, num_layer, emb_dim, head, class_num, dim_expan = 4, dropout = 0):
    super(ViTransformer,self).__init__() 
    self.num_layer = num_layer
    self.emb_dim = emb_dim
    self.patch = Patch(img_size, patch_num, emb_dim) #img_size, patch_num, emb_dim
    self.pos_emb = Positional_Emb(img_size, patch_num, emb_dim, dropout= dropout) #img_size, patch_num, emb_dim, dropout
    self.encoder = Encoder(num_layer, emb_dim, head, dim_expan = dim_expan, dropout = dropout) # num_layer, seq_len, emb_dim, head
    self.linear = torch.nn.Linear(emb_dim, class_num) #,bias = False

  def forward(self, x):
    x = self.pos_emb(self.patch(x))
    out = self.encoder(x)
    out = self.linear(out[:,0,:]) 
    return out
    
