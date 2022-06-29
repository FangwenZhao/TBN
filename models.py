import math

import numpy as np
import torch
import torch.nn as nn


from transfer_losses import TransferLoss
import backbones


class TransferNet(nn.Module):
    def __init__(self, num_class, base_net='resnet50', transfer_loss='mmd',  bottleneck_width=256, max_iter=1000, kernel=16,**kwargs):
        super(TransferNet, self).__init__()
        self.num_class = num_class
        self.base_network = backbones.get_backbone(base_net)

        self.transfer_loss = transfer_loss
        bottleneck_list = [
            nn.Linear(self.base_network.output_num(), bottleneck_width),
            nn.ReLU(),
            ]
        bottleneck_list1 = [
            nn.Linear(self.base_network.output_num(), bottleneck_width),
            nn.ReLU(),
        ]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        self.bottleneck_layer1 = nn.Sequential(*bottleneck_list1)

        self.source_classifier_layer =  nn.Linear(bottleneck_width, num_class)
        self.target_classifier_layer =  nn.Linear(bottleneck_width, num_class)


        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.Criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def target_clusters(self,feature, labels,num_class, threshold=0.8):

        label_value = labels[1]
        label_conf = labels[0]
        batch_size = label_conf.nelement()

        mask_conf_sum = math.floor(threshold * batch_size)
        mask_conf = label_conf>torch.topk(label_conf, mask_conf_sum)[0][...,-1,None]

        # 计算样本之间的L2距离
        total0 = feature.unsqueeze(0).expand(
            int(feature.size(0)), int(feature.size(0)), int(feature.size(1)))  # 64，64，256  64个total
        total1 = feature.unsqueeze(1).expand(
            int(feature.size(0)), int(feature.size(0)),
            int(feature.size(1)))  # 64,64,256
        L2_distance = torch.tensor(total0 - total1, dtype=torch.float)
        L2_distance = torch.norm(L2_distance, p="fro", dim=2)
        L2_distance.masked_fill_(~mask_conf, 0)  # 对低置信度的样本的距离使用0进行填充

        conf_tmp = label_conf.unsqueeze(1)
        conf_mat = conf_tmp @ conf_tmp.t()
        L2_distance = torch.mul(L2_distance, conf_mat)
        label_conf.masked_fill_(~mask_conf, 0)
        distances = torch.zeros([batch_size, num_class],device='cuda')
        class_counts = torch.zeros([num_class], device='cuda')
        for i in range(num_class):
            for j in range(batch_size):
                if i == label_value[j]:
                    distances[:, i] += L2_distance[:, j]  # 相同类别的样本之间的距离相加
                    class_counts[i] += label_conf[j]

        mask_counts = torch.nonzero(class_counts).squeeze(1)

        class_counts[class_counts == 0] = 100
        distances = distances / class_counts  # 去掉预测的某类样本过多导致距离之和变大的影响
        distances = torch.index_select(distances, dim=1, index=mask_counts)  # 选出该批次中 符合要求的类别
        labels = torch.argmin(distances, 1)
        labels_tar = mask_counts[labels]

        return labels_tar

    def forward(self, source, target, labels,threshold=1):
        if source==None:
            target_feature = self.bottleneck_layer1(target)
            labels_ = self.target_clusters(target_feature, labels, self.num_class,threshold)
            labels_post=self.target_classifier_layer(target_feature)

            clf_loss = self.criterion(labels_post, labels_)
            labels_post = torch.nn.functional.softmax(labels_post, dim=1).detach()
            labels_post = labels_post.max(1)[1]

            return labels_post, labels_,clf_loss
        source_f = self.base_network(source)
        target_f = self.base_network(target)
        source_b = self.bottleneck_layer(source_f)
        target_b = self.bottleneck_layer(target_f)
        # transfer
        kwargs = {}

        source_clf = self.source_classifier_layer(source_b)
        target_label = self.source_classifier_layer(target_b)
        target_label = torch.nn.functional.softmax(target_label, dim=1).detach()
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = labels
            kwargs['target_logits'] = target_label
        transfer_loss = self.adapt_loss(source_b, target_b, **kwargs)

        # classification
        target_label = target_label.max(1)
        source_loss = self.criterion(source_clf, labels)+transfer_loss

        return source_loss,target_f.detach(), target_label


    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr},
            {'params': self.bottleneck_layer1.parameters(), 'lr': 1.0 * initial_lr},
            {'params': self.source_classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
            {'params': self.target_classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        x = self.base_network(x)
        x = self.bottleneck_layer1(x)
        clf = self.target_classifier_layer(x)
        # x = self.bottleneck_layer(x)
        # clf = self.source_classifier_layer(x)
        return clf


    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass