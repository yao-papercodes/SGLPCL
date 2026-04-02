import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import SAGEConv, TransformerConv, GATConv, SAGPooling
from OT_torch_ import cost_matrix_batch_torch, GW_distance_uniform, IPOT_distance_torch_batch_uniform
import math
from torch_geometric.data import Data
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'model'))
from backbone import ResNet50Fc as ResNet50
from utils_HSI import *
from utils_PL import *
import torch.nn.init as init

class vgg16(nn.Module):
    def __init__(self, in_channels, num_classes, patch_size, init_weights=True, batch_norm=True):
        super(vgg16, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        cfg = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [32, 32, 64, 64],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
            'yao': [64, 128, 256, 256]
        }
        layers = []
        for v in cfg['D']:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, stride=1, kernel_size=3)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.LeakyReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*layers)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.in_channels,
                             self.patch_size, self.patch_size))
            x = self.features(x)
            t, c, w, h = x.size()
        return t * c * w * h

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # raw x = (bs, c, ps, ps)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class PredictorCNN(nn.Module):
    def __init__(self, in_dim=1024, num_class=7, prob=0.5, lambd=1, init_type='kaiming_normal'):
        super(PredictorCNN, self).__init__()
        self.fc1 = nn.Linear(in_dim, 4096)
        self.bn1_fc = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 256)
        self.bn2_fc = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_class)
        self.bn_fc3 = nn.BatchNorm1d(num_class)
        self.prob = prob
        self.init_type = init_type
        self.lambd = lambd
        self._initialize_weights()

    def set_lambda(self, lambd):
        self.lambd = lambd
    
    def _initialize_weights(self):
        if self.init_type == 'kaiming_normal':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif self.init_type == 'kaiming_uniform':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        elif self.init_type == 'xavier_normal':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
        elif self.init_type == 'xavier_uniform':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.fc3(x)
        return x

class PredictorGCN(nn.Module):
    def __init__(self, in_dim,num_class,dropout=0.5, lambd=1, init_type='kaiming_normal'):
        super(PredictorGCN, self).__init__()
        self.fc1 = nn.Linear(in_dim, 1024, bias=True)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, 1024, bias=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(1024, num_class, bias=True)
        self.lambd = lambd
        self.init_type = init_type
        self._initialize_weights()

    def set_lambda(self, lambd):
        self.lambd = lambd
    
    def _initialize_weights(self):
        if self.init_type == 'kaiming_normal':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif self.init_type == 'kaiming_uniform':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        elif self.init_type == 'xavier_normal':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
        elif self.init_type == 'xavier_uniform':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x)
        return x

class Topology_Extraction(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Topology_Extraction, self).__init__()
        # self.conv1 = SAGEConv(in_channels, hidden_channels)
        # self.bn1 = nn.BatchNorm1d(hidden_channels)
        # self.conv2 = SAGEConv(hidden_channels, out_channels)
        # self.bn2 = nn.BatchNorm1d(out_channels)

        # self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads)
        # self.bn1 = nn.BatchNorm1d(hidden_channels*heads)
        # self.conv2 = TransformerConv(hidden_channels*heads, int(out_channels/heads), heads=heads)
        # self.bn2 = nn.BatchNorm1d(out_channels)

        # self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.5)
        # self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        # self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=0.5)
        # self.bn2 = nn.BatchNorm1d(out_channels * self.conv2.heads)

        self.conv = GATConv(in_channels, out_channels, heads=16, concat=True, dropout=0.5)
        self.final_out_channels = out_channels * self.conv.heads if self.conv.concat else out_channels
        self.bn = nn.BatchNorm1d(self.final_out_channels)

    def embed_pos(self, x, y, d_model=64):
        device = x.device
        bs = x.size(0)
        x, y = x.to(dtype=torch.float32), y.to(dtype=torch.float32)
        div_term = torch.exp(torch.arange(0, int(d_model/2), 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)).to(device)
        embedding = torch.zeros((bs, d_model)).to(device)
        embedding[:, 0:int(d_model/2):2] = torch.sin(x.unsqueeze(1) * div_term)
        embedding[:, 1:int(d_model/2):2] = torch.cos(x.unsqueeze(1) * div_term)
        embedding[:, int(d_model/2)::2] = torch.sin(y.unsqueeze(1) * div_term)
        embedding[:, int(d_model/2)+1::2] = torch.cos(y.unsqueeze(1) * div_term)
        return embedding

    def embed_psk(self, psk):
        labels = psk['data']
        device = labels.device
        if psk['flag'] == 'src':
            labels = trans_one_hot(labels, psk['classes'])
            
        probs, entropy = cal_entropy(labels, psk['flag'])
        # return torch.concat((probs, entropy), dim=1)
        return probs * entropy

    def forward(self, data, idxs=None, psk=None):
        x, edge_index = data.x, data.edge_index
        if idxs != None:
            pos_embedding = self.embed_pos(idxs['data'][0], idxs['data'][1], d_model=idxs['dims'])
            x = torch.concat((x, pos_embedding), dim=1)
        if psk != None:
            psk_embedding = self.embed_psk(psk)
            x = torch.concat((x, psk_embedding), dim=1)
        
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        return x

class ClassAligment(nn.Module):
    def __init__(self, num_classes, feature_dim, decay=0.90):
        super(ClassAligment, self).__init__()
        self.source_moving_centroid = nn.Parameter(torch.zeros(num_classes, feature_dim), requires_grad=False)
        self.target_moving_centroid = nn.Parameter(torch.zeros(num_classes, feature_dim), requires_grad=False)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.decay = decay
        self.tempreture = 0.000001
        self.tmp_source_centroid = None
        self.tmp_target_centroid = None

    def triplet_loss(self, src_prototypes, tar_prototypes, margin=1.0):
        loss = torch.tensor([0.]).to(margin.device)
        for i in range(self.num_classes):
            # 计算正样本对距离
            if i in src_prototypes.keys() and i in tar_prototypes.keys():
                pos_distance = 1 - F.cosine_similarity(src_prototypes[i], tar_prototypes[i], dim=-1)
            else:
                continue

            # 计算负样本对距离
            neg_distances = []
            for j in range(self.num_classes):
                if j != i:
                    if j in src_prototypes.keys():
                        neg_distances.append(1 - F.cosine_similarity(src_prototypes[i], src_prototypes[j], dim=-1))
                    # if j in tar_prototypes.keys():
                    #     neg_distances.append(torch.norm(src_prototypes[i] - tar_prototypes[j], p=2))
            neg_distance = torch.mean(torch.stack(neg_distances))
            
            # print(f"class {i}: pos={pos_distance.item():.4f}, neg={neg_distance.item():.4f}, margin={margin.item():.4f}")

            # 计算 Triplet Loss
            triplet_loss = torch.clamp(margin + pos_distance - neg_distance, min=0)
            loss += triplet_loss

        return loss / self.num_classes  # 可以根据需要调整损失的归一化方式

    def contrastive_loss(self, src_prototypes, tar_prototypes, weight):
        device = src_prototypes[list(src_prototypes.keys())[0]].device
        labels = []
        src_prototypes_, tar_prototypes_ = None, None
        idxs = []
        for i in range(self.num_classes):
            if i in src_prototypes.keys() and i in tar_prototypes.keys():
                if src_prototypes_ == None:
                    src_prototypes_, tar_prototypes_ = src_prototypes[i], tar_prototypes[i]
                else:
                    src_prototypes_ = torch.concat((src_prototypes_, src_prototypes[i]), dim=0)
                    tar_prototypes_ = torch.concat((tar_prototypes_, tar_prototypes[i]), dim=0)
                idxs.append(i)
                labels.append(len(labels))
        src_protos = F.normalize(src_prototypes_, dim=1)
        tar_protos = F.normalize(tar_prototypes_, dim=1)
        labels = torch.tensor(labels).to(device)
        sim_matrix = torch.matmul(src_protos, tar_protos.T)
        idxs = torch.tensor(idxs).to(device)
        weight_ = weight[idxs][:, idxs]
        weight_0 = F.softmax(weight_, dim=0) + self.tempreture
        weight_1 = F.softmax(weight_, dim=1) + self.tempreture
        try:
            sim_matrix = ( sim_matrix * weight_0 + sim_matrix * weight_1 ) / 2
        except Exception as e:
            pass
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
        

    def forward(self, source, target, src_labels, tar_labels, weigth):
        device = source.device
        # Computing prototypes for each class as the mean of the extracted features
        source_dict = {}
        target_dict = {}
        source_dict = dict(zip(source, src_labels))
        target_dict = dict(zip(target, tar_labels))

        current_source_dict, current_target_dict = {}, {}
        final_source_dict, final_target_dict = {}, {}

        # print(f"source_dict = {source_dict}")

        # Compute prototypes for source domain
        for key, label in source_dict.items():
            if label.item() not in current_source_dict:
                current_source_dict[label.item()] = []
            current_source_dict[label.item()].append(key)

        # Compute prototypes for target domain
        for key, label in target_dict.items():
            if label.item() not in current_target_dict:
                current_target_dict[label.item()] = []
            current_target_dict[label.item()].append(key)
        

        for label, source_prototypes in current_source_dict.items():
            source_prototypes = torch.mean(torch.stack(source_prototypes), dim=0, keepdim=True)
            final_source_dict[int(label)] = self.decay * self.source_moving_centroid[int(label)] + (1-self.decay) * source_prototypes
        
        for label, target_prototypes in current_target_dict.items():
            target_prototypes = torch.mean(torch.stack(target_prototypes), dim=0, keepdim=True)
            target_prototypes_norm = F.normalize(target_prototypes, dim=1)
            source_prototypes_norm = F.normalize(self.source_moving_centroid[int(label)], dim=0)
            final_target_dict[int(label)] = (1 - self.decay) * source_prototypes_norm + self.decay * target_prototypes_norm
        
        # dist_loss = self.triplet_loss(final_source_dict, final_target_dict, margin=compute_mmd(source, target))
        dist_loss = self.contrastive_loss(final_source_dict, final_target_dict, weigth)
        
        self.tmp_source_centroid = final_source_dict
        self.tmp_target_centroid = final_target_dict
        self.update_centroid()
        
        return dist_loss

    @torch.no_grad()
    def update_centroid(self):
        for key, value in self.tmp_source_centroid.items():
            self.source_moving_centroid.data[key] = value
        # for key, value in self.tmp_target_centroid.items():
        #     self.target_moving_centroid.data[key] = value

class ProjectHead(nn.Module):
    def __init__(self, cnn_in_channels, gcn_in_channels, out_channels=128):
        super(ProjectHead, self).__init__()
        self.cnn_proj = nn.Linear(cnn_in_channels, out_channels, bias=True)
        self.gcn_proj = nn.Linear(gcn_in_channels, out_channels, bias=True)
        self.out_channels = out_channels

    def forward(self, x, flag='cnn'):
        if flag == 'cnn':
            x = self.cnn_proj(x)
        elif flag == 'gcn':
            x = self.gcn_proj(x)
        return x
    
class ClassifyHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=7, dropout_prob=0.5):
        super(ClassifyHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

class GraphMCD(nn.Module):
    def __init__(self, in_channels, num_classes, patch_size):
        super(GraphMCD, self).__init__()
        self.classes = num_classes
        self.pos_dims = 64
        #@ CNN (backbone / generator / feature extractor)
        self.backbone = vgg16(in_channels, num_classes, patch_size)
        self.backbone_output_dim = self.backbone._get_final_flattened_size()
        #@ GCN layer
        self.intra_gcn = Topology_Extraction(self.backbone_output_dim + self.pos_dims + self.classes, 64, 32)
        self.gcn_output_dim = self.intra_gcn.final_out_channels
        #@ Project layer
        self.project = ProjectHead(self.backbone_output_dim, self.gcn_output_dim)
        self.proj_output_dim = self.project.out_channels
        #@ MLP layer
        self.classify = ClassifyHead(input_dim=self.proj_output_dim, output_dim=num_classes)
        #@ class alignment module
        self.ca_cnn = ClassAligment(self.classes, self.proj_output_dim)
        self.ca_intra_gcn = ClassAligment(self.classes, self.proj_output_dim)
    
    @torch.no_grad()
    def refining_pseudo_labels(self, tar_cnn_feats, tar_gcn_feats, banks, topk=3):
        
        plabels_by_tar_cnn, confi_by_tar_cnn, dists_by_tar_cnn = \
            get_plabels_confi_dist(tar_cnn_feats, banks['tar']['cnn_feats'], banks['tar']['cnn_probs'], topk)
        plabels_by_tar_intra_gcn, confi_by_tar_intra_gcn, dists_by_tar_intra_gcn = \
            get_plabels_confi_dist(tar_gcn_feats, banks['tar']['gcn_feats'], banks['tar']['gcn_probs'], topk)
        
        plabels_by_src_cnn, confi_by_src_cnn, dists_by_src_cnn = \
            get_plabels_confi_dist(tar_cnn_feats, banks['src']['cnn_feats'], banks['src']['cnn_probs'], topk)
        plabels_by_src_intra_gcn, confi_by_src_intra_gcn, dists_by_src_intra_gcn = \
            get_plabels_confi_dist(tar_gcn_feats, banks['src']['gcn_feats'], banks['src']['gcn_probs'], topk)
        
        domain_mmd_cnn = compute_mmd(banks['src']['cnn_feats'], banks['tar']['cnn_feats'])
        domain_mmd_gcn_intra = compute_mmd(banks['src']['gcn_feats'], banks['tar']['gcn_feats'])
        
        mask_cnn = ((dists_by_src_cnn + domain_mmd_cnn) < dists_by_tar_cnn).type(torch.int64) # (bs, 1)
        pseudo_labels_cnn_con = torch.concat((plabels_by_tar_cnn.unsqueeze(1), plabels_by_src_cnn.unsqueeze(1)), dim=1) # (bs, 2)
        pseudo_labels_cnn = torch.gather(pseudo_labels_cnn_con, 1, mask_cnn).squeeze(1) # (bs, )
        confi_cnn_con = torch.concat((confi_by_tar_cnn.unsqueeze(1), confi_by_src_cnn.unsqueeze(1)), dim=1) # (bs, 2)
        confi_cnn = torch.gather(confi_cnn_con, 1, mask_cnn).squeeze(1) # (bs, )
        
        mask_intra_gcn = ((dists_by_src_intra_gcn + domain_mmd_gcn_intra) < dists_by_tar_intra_gcn).type(torch.int64) # (bs, 1)
        pseudo_labels_intra_gcn_con = torch.concat((plabels_by_tar_intra_gcn.unsqueeze(1), plabels_by_src_intra_gcn.unsqueeze(1)), dim=1) # (bs, 2)
        pseudo_labels_intra_gcn = torch.gather(pseudo_labels_intra_gcn_con, 1, mask_intra_gcn).squeeze(1) # (bs, )
        confi_intra_gcn_con = torch.concat((confi_by_tar_intra_gcn.unsqueeze(1), confi_by_src_intra_gcn.unsqueeze(1)), dim=1) # (bs, 2)
        confi_intra_gcn = torch.gather(confi_intra_gcn_con, 1, mask_intra_gcn).squeeze(1) # (bs, )

        return pseudo_labels_cnn, confi_cnn, pseudo_labels_intra_gcn, confi_intra_gcn

    @torch.no_grad()
    def accessing_statics_similarty(self, feats_dict, banks, topk=3):
        device = feats_dict['src']['cnn'].device
        cnn_statics, gcn_statics = torch.zeros(self.classes, self.classes).to(device), torch.zeros(self.classes, self.classes).to(device)
        d_l_dict = dict([(f'{f_type}_{k_feat}_2_{v_feat}', get_dist_label(feats_dict[k_feat][f_type], banks[v_feat][f_type+'_feats'], banks[v_feat][f_type+'_probs'], banks[v_feat]['gt_labels'], topk=topk)) \
                for f_type in ['cnn', 'gcn'] for k_feat in ['src', 'tar'] for v_feat in ['src', 'tar'] ])
        mmd_dict = dict([(f_type, compute_mmd(banks['src'][f_type+'_feats'], banks['tar'][f_type+'_feats'])) for f_type in ['cnn', 'gcn']])
        
        mask_cnn_tar_close2_src = (d_l_dict['cnn_tar_2_src'][0] + mmd_dict['cnn'] < d_l_dict['cnn_tar_2_tar'][0]).squeeze(1)
        mask_cnn_src_close2_tar = (d_l_dict['cnn_src_2_tar'][0] + mmd_dict['cnn'] < d_l_dict['cnn_src_2_src'][0]).squeeze(1)
        mask_gcn_tar_close2_src = (d_l_dict['gcn_tar_2_src'][0] + mmd_dict['gcn'] < d_l_dict['gcn_tar_2_tar'][0]).squeeze(1)
        mask_gcn_src_close2_tar = (d_l_dict['gcn_src_2_tar'][0] + mmd_dict['gcn'] < d_l_dict['gcn_src_2_src'][0]).squeeze(1)
        
        # 计算cnn_statics
        logits_cnn_tar_close2_src = self.classify(feats_dict['tar']['cnn'][mask_cnn_tar_close2_src])
        self_labels = logits_cnn_tar_close2_src.max(1)[1] # 目标域样本自身的标签信息
        banks_labels = d_l_dict['cnn_tar_2_src'][2][mask_cnn_tar_close2_src] # 跟目标域样本靠近的那些bank中的源域真实标签信息
        for s, t in zip(banks_labels, self_labels):
            cnn_statics[s][t] += 1

        self_labels = feats_dict['src']['labels'][mask_cnn_src_close2_tar].long()
        banks_labels = d_l_dict['cnn_src_2_tar'][1][mask_cnn_src_close2_tar]
        for s, t in zip(self_labels, banks_labels):
            cnn_statics[s][t] += 1
            
        # 计算gcn_statics
        logits_gcn_tar_close2_src = self.classify(feats_dict['tar']['gcn'][mask_gcn_tar_close2_src])
        self_labels = logits_gcn_tar_close2_src.max(1)[1] # 目标域样本自身的标签信息
        banks_labels = d_l_dict['gcn_tar_2_src'][2][mask_gcn_tar_close2_src] # 跟目标域样本靠近的那些bank中的源域真实标签信息
        for s, t in zip(banks_labels, self_labels):
            gcn_statics[s][t] += 1

        self_labels = feats_dict['src']['labels'][mask_gcn_src_close2_tar].long()
        banks_labels = d_l_dict['gcn_src_2_tar'][1][mask_gcn_src_close2_tar]
        for s, t in zip(self_labels, banks_labels):
            gcn_statics[s][t] += 1
        
        return cnn_statics, gcn_statics


    def forward(self, source, target=None, src_labels=None, idxs=None, banks=None):
        bs = source.shape[0]
        out = self.backbone(source)
        cnn_feats = self.project(out)
        cnn_logits = self.classify(cnn_feats)
        
        if not self.training:
            return cnn_logits
        else:
            src_cnn_out, src_cnn_feats, src_cnn_logits = out, cnn_feats, cnn_logits
            src_share_graph = getGraphdataOneDomain(src_cnn_out, bs)
            src_pos = {'flag': 'src', 'data': [idxs['x_src'], idxs['y_src']], 'dims': self.pos_dims}
            src_psk = {'flag': 'src', 'data': src_labels, 'classes': self.classes}
            src_intra_gcn_out = self.intra_gcn(src_share_graph, src_pos, src_psk)
            src_gcn_feats = self.project(src_intra_gcn_out, flag='gcn')
            src_intra_gcn_logits = self.classify(src_gcn_feats)

            tar_cnn_out = self.backbone(target)
            tar_cnn_feats = self.project(tar_cnn_out)
            tar_cnn_logits = self.classify(tar_cnn_feats)
            tar_cnn_logits_ = tar_cnn_logits.clone().detach()
            tar_share_graph = getGraphdataOneDomain(tar_cnn_out, bs)
            tar_pos = {'flag': 'tar', 'data': [idxs['x_tar'], idxs['y_tar']], 'dims': self.pos_dims}
            tar_psk = {'flag': 'tar', 'data': tar_cnn_logits_, 'classes': self.classes}
            tar_intra_gcn_out = self.intra_gcn(tar_share_graph, tar_pos, tar_psk)
            tar_gcn_feats = self.project(tar_intra_gcn_out, flag='gcn')
            tar_intra_gcn_logits = self.classify(tar_gcn_feats)

            if banks['src'] != None:
                cnn_statics, gcn_statics = self.accessing_statics_similarty({'src': {'cnn': src_cnn_feats, 'gcn': src_gcn_feats, 'labels': src_labels}, \
                    'tar': {'cnn': tar_cnn_feats, 'gcn': tar_gcn_feats}}, banks, topk=3)
                pseudo_labels_cnn, confi_cnn, pseudo_labels_intra_gcn, confi_intra_gcn = \
                    self.refining_pseudo_labels(tar_cnn_feats, tar_gcn_feats, banks, topk=3)

                dist_loss = self.ca_cnn(src_cnn_feats, tar_cnn_feats, src_labels, pseudo_labels_cnn, cnn_statics)
                dist_loss += self.ca_intra_gcn(src_gcn_feats, tar_gcn_feats, src_labels, pseudo_labels_intra_gcn, gcn_statics)
                dist_loss /= 2
            
            else:
                pseudo_labels_cnn, confi_cnn, pseudo_labels_intra_gcn, confi_intra_gcn, dist_loss = [None] * 5

            return src_cnn_feats, src_cnn_logits, src_gcn_feats, src_intra_gcn_logits, \
                tar_cnn_feats, tar_cnn_logits, tar_gcn_feats, tar_intra_gcn_logits, \
                pseudo_labels_cnn, confi_cnn, pseudo_labels_intra_gcn, confi_intra_gcn, dist_loss

