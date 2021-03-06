# -*- coding: utf-8 -*-
"""
@author: mehme
"""

import torch
import torch.nn as nn
from torchvision import models
#from train import label2target


label2target = {'balloon':1, 'background':0}
target2label = {t:l for l,t in label2target.items()}
background_class = label2target['background']
    
class RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        feature_dims = 25088
        
        vgg_backbone = models.vgg16(pretrained=True)
        vgg_backbone.classifier = nn.Sequential() 
        #backbone'un son kısmında yer alan sınıflandırma kısmını bozarak sequential bir hale getiriyoruz.
        for param in vgg_backbone.parameters():
            param.required_grad = False
        
        self.backbone = vgg_backbone
        self.cls_score = nn.Linear(feature_dims, len(label2target))
        # class sayımız kadar çıktı verecek bir linear layer
        self.bound_box = nn.Sequential(
            nn.Linear(feature_dims,512),
            nn.ReLU(),
            nn.Linear(512,4),
            nn.Tanh()
            )
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.L1Loss()
        
    def forward(self,input):
        scores = torch.zeros(input.shape[0], 2).cuda()
        bound_boxes = torch.zeros(input.shape[0], 4).cuda()
        
        for i in range(0, len(input),32):
            inputs = input[i:i+32]
            feat = self.backbone(inputs)
            cls_score = self.cls_score(feat)
            bound_box = self.bound_box(feat)
            
            scores[i:i+32] = cls_score
            bound_boxes[i:i+32] = bound_box
            
        return scores, bound_boxes
    
    def loss_calc(self, probs, _deltas, labels, deltas):
        detection_loss = self.cel(probs,labels)
        ixs, = torch.where(labels != label2target["background"])
        _deltas = _deltas[ixs]
        deltas = deltas[ixs]
        self.lmb = 10.0
        if len(ixs)>0:
            regression_loss = self.sl1(_deltas, deltas)
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss.detach()
        else:
            regression_loss = 0
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss
