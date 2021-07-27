# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb
class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(ContrastiveLoss, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, features, labels):
        pdb.set_trace()
        feature_size_h = 64
        feature_size_w = 32
        features = F.interpolate(features, size=(feature_size_w, feature_size_h), mode='bilinear')
        labels = labels.type(torch.cuda.FloatTensor)
        labels = torch.unsqueeze(labels, 1)
        labels = F.interpolate(labels,  size=(feature_size_w, feature_size_h), mode='nearest')
        labels = torch.squeeze(labels)
        features = features.view(features.size(0),features.size(1),-1)
        features = torch.transpose(features, 1,2)        
        features = F.normalize(features,p=2, dim=2)

        labels = labels.view(labels.size(0),-1,1)
        labels = labels.repeat((1,1,labels.size(1)))
        labels_t = torch.transpose(labels, 1,2)

        weight_mask = torch.eq(labels, labels_t)
        weight_mask = weight_mask.type(torch.cuda.FloatTensor)
        ignore_mask = torch.eq(labels, self.ignore_label)
        ignore_mask = ignore_mask.type(torch.cuda.FloatTensor)
        ignore_mask_t = torch.eq(labels_t, self.ignore_label)
        ignore_mask_t = ignore_mask_t.type(torch.cuda.FloatTensor)
         
        weight_mask = weight_mask*(1-ignore_mask) 
        
        
        logit_sum = torch.sum(features)
        
        if torch.isnan(logit_sum):
            pdb.set_trace()
        
        logits_ab = torch.matmul(features,torch.transpose(features,1,2))/0.07
        

        loggits_ab = logits_ab*(1-ignore_mask_t)+ignore_mask_t*(-1e9)
        loss = self.supervised_contrastive_loss(logits_ab, weight_mask, logits_ab.size(1), ignore_mask_t)



        return loss


    def supervised_contrastive_loss(self, logits, weight_mask, pixel_size, ignore_mask): 

        #final_logits = torch.exp(logits)
        final_logits = F.softmax(logits,dim=2)
        
        #neg_mask = 1 - ignore_mask - weight_mask
        #neg_logits = final_logits*neg_mask
        #neg_sum = torch.sum(neg_logits, 2, keepdim = True)
        #final_logits = torch.div(final_logits,neg_sum)
        #final_logits = final_logits*(1-torch.isnan(final_logits))
        #final_logits[final_logits != final_logits] = 0        
        masks = F.one_hot(torch.arange(0, pixel_size),pixel_size)
        masks = torch.unsqueeze(masks, 0)
        masks = masks.type(torch.cuda.FloatTensor)
        final_logits = final_logits*(1-ignore_mask) + ignore_mask
        log_likelihood = -torch.log(final_logits)
        #log_likelihood = log_likelihood - log_likelihood*masks

        log_likelihood = log_likelihood*weight_mask
        log_likelihood = torch.sum(log_likelihood,2)
        den = torch.sum(weight_mask,2)
        ind = torch.eq(den,0)
        ind = ind.type(torch.cuda.FloatTensor)
        den = den*(1-ind) + ind
        log_likelihood = log_likelihood / den
        log_likelihood[log_likelihood != log_likelihood] = 0
        log_likelihood1 = log_likelihood 
        index = 1 - ind
        index_sum = torch.sum(index,1)
        ind2 = torch.eq(index_sum,0)
        ind2 = ind2.type(torch.cuda.FloatTensor)

        index_sum = index_sum + ind2
        log_likelihood = torch.sum(log_likelihood,1)/index_sum
        loss = torch.mean(log_likelihood)
        if  torch.eq(loss,0):
            pdb.set_trace()
        
        nan_mask = torch.isnan(loss)
        if nan_mask:
            pdb.set_trace()
        
        return loss





class OhemCrossEntropy(nn.Module): 
    def __init__(self, ignore_label=-1, thres=0.7, 
        min_kept=100000, weight=None): 
        super(OhemCrossEntropy, self).__init__() 
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label 
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label, 
                                             reduction='none') 
    
    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label         
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_label] = 0 
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)] 
        threshold = max(min_value, self.thresh) 
        
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold] 
        return pixel_losses.mean()
