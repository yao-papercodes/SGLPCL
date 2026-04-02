from __future__ import print_function
import argparse
import os
# os.environ['cuda_visible_devices'] = '0,1'
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from model import GMCD
import numpy as np
from utils_HSI import *
from utils_PL import *
from datasets import get_dataset, HyperX, data_prefetcher
from datetime import datetime
from collections import Counter
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import scipy.io as io
from sklearn.metrics import classification_report
import torch.nn as nn
import datetime
import shutil
import pickle
from tqdm import tqdm, trange
import debugpy
import gc


def construct_argument():
    parser = argparse.ArgumentParser(description='PyTorch')

    #@ Output setting
    parser.add_argument('--save_path', type=str, default="./results/",
                        help='the path to save the model')
    parser.add_argument('--data_path', type=str, default='../Datasets/Houston/',
                        help='the path to load the data')
    parser.add_argument('--log_path', type=str, default='./logs',
                        help='the path to load the data')
    parser.add_argument('--output_path', type=str, default='./exp',
                        help='the path to save this exp data.')

    #@ Training setting (Default)
    parser.add_argument('--source_name', type=str, default='Dioni',
                        help='the name of the source dir, can automaticly change by programe')
    parser.add_argument('--target_name', type=str, default='Loukia',
                        help='the name of the test dir, can automaticly change by programe')
    parser.add_argument('--cuda', type=int, default=0,
                        help="Specify CUDA device (defaults to -1, which learns on CPU)")
    group_train = parser.add_argument_group('Training')
    group_train.add_argument('--patch_size', type=int, default=12,
                        help="Size of the spatial neighbourhood (optional, if "
                        "absent will be set by the model)")
    group_train.add_argument('--lr', type=float, default=1e-2,
                        help="Learning rate, set by the model if not specified.")
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    group_train.add_argument('--batch_size', type=int, default=100,
                        help="Batch size (optional, if absent will be set by the model")
    group_train.add_argument('--test_stride', type=int, default=1,
                        help="Sliding window step stride during inference (default = 1)")
    parser.add_argument('--seed', type=int, default=1233, metavar='S',
                        help='random seed ')
    parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--l2_decay', type=float, default=1e-4,
                        help='the L2 weight decay')
    parser.add_argument('--num_epoch', type=int, default=150,
                        help='the number of epoch')
    parser.add_argument('--num_trials', type=int, default=10, help='the number of epoch')
    parser.add_argument('--training_sample_ratio', type=float, default=0.05, help='training sample ratio')
    parser.add_argument('--re_ratio', type=int, default=10,
                        help='multiple of data augmentation. It can be calculated automatically during the training process, and is not predetermined!')
    parser.add_argument('--checkpoint', type=str, default='None', help='checkpoint path')
    parser.add_argument('--isTest', type=bool, default=False, help='whether is just concede the inference stage')
    parser.add_argument('--classNum', type=int, default=7, help='class number, no need to predefine')

    #! Hyperparameters
    parser.add_argument('--coef_cls_tar', type=float, default=1.0, help='coefficient of discrepency')

    # Data augmentation parameters
    group_da = parser.add_argument_group('Data augmentation')
    group_da.add_argument('--flip_augmentation', action='store_true', default=True,
                        help="Random flips (if patch_size > 1)")
    group_da.add_argument('--rotate_augmentation', action='store_true', default=True,
                        help="Random rotate (if patch_size > 1)")
    group_da.add_argument('--radiation_augmentation', action='store_true',default=True,
                        help="Random radiation noise (illumination)")
    group_da.add_argument('--mixture_augmentation', action='store_true',default=False,
                        help="Random mixes between spectra")

    args = parser.parse_args()
    return args

def train(epoch, model_GMCD, num_epoch):

    #@ Build the optimizers
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch) / num_epoch), 0.75)
    # optimizer_GMCD = optim.SGD(model_GMCD.parameters(), lr=LEARNING_RATE, momentum=args.momentum,weight_decay = args.l2_decay)
    optimizer_GMCD = torch.optim.Adam(model_GMCD.parameters(), lr=LEARNING_RATE)
    if (epoch-1)%10 == 0:
        print('learning rate{: .4f}'.format(LEARNING_RATE))

    #@ Initialize the metrics
    global writer
    Loss_Cls, Loss_Cls_CNN, Loss_Cls_GCN_Intra, Loss_Cls_Tar, Loss_Dist = 0, 0, 0, 0, 0
    Correct_Src_CNN, Correct_Src_GCN_Intra, Correct_Src_GCN_Inter,\
        Correct_Tar_CNN, Correct_Tar_GCN_Intra, Correct_Tar_GCN_Inter = 0, 0, 0, 0, 0, 0
    Correct_PL_CNN, Correct_PL_Intra_GCN = torch.tensor(0.), torch.tensor(0.)
    len_tar_temp = 0

    #@ Bank variables initialize
    src_cnn_logits_list, src_cnn_feats_list, src_gcn_intra_logits_list, src_gcn_intra_feats_list, src_labels_list = [], [], [], [], []
    tar_cnn_logits_list, tar_cnn_feats_list, tar_gcn_intra_logits_list, tar_gcn_intra_feats_list, tar_labels_list = [], [], [], [], []
    tar_stop = False

    iter_source = data_prefetcher(train_loader)
    iter_target = data_prefetcher(train_tar_loader)
    num_iter = len_src_loader
    bs = train_loader.batch_size

    model_GMCD.train()
    qbar = tqdm(range(1, num_iter), colour='#79a0c9')
    for i in qbar:

        if 0 < (len_tar_train_dataset-i*bs) < bs or i % len_tar_train_loader == 0:
            iter_target = data_prefetcher(train_tar_loader)
            tar_stop = True
        index_src, data_src, label_src, x_src, y_src = iter_source.next()
        index_tar, data_tar, label_tar, x_tar, y_tar = iter_target.next()
        label_src = label_src - 1
        label_tar = label_tar - 1
        idxs = {
            'x_src': x_src,
            'y_src': y_src,
            'x_tar': x_tar,
            'y_tar': y_tar
        }
        
        optimizer_GMCD.zero_grad()
        src_cnn_feats, src_cnn_logits, src_gcn_intra_feats, src_gcn_intra_logits, \
        tar_cnn_feats, tar_cnn_logits, tar_gcn_intra_feats, tar_gcn_intra_logits, \
        pseudo_labels_cnn, confi_cnn, pseudo_labels_intra_gcn, confi_intra_gcn, \
        loss_dist = model_GMCD(data_src, data_tar, label_src, idxs, banks=banks)
        
        loss_cls_cnn = F.nll_loss(F.log_softmax(src_cnn_logits, dim=1), label_src.long())
        loss_cls_gcn_intra = F.nll_loss(F.log_softmax(src_gcn_intra_logits, dim=1), label_src.long())
        loss_cls_src = loss_cls_cnn + loss_cls_gcn_intra
        if pseudo_labels_cnn != None:
            # loss_cls_tar_cnn = F.nll_loss(F.log_softmax(tar_cnn_logits, dim=1), pseudo_labels_cnn.long(), reduction='none')
            # loss_cls_tar_intra_gcn = F.nll_loss(F.log_softmax(tar_gcn_intra_logits, dim=1), pseudo_labels_intra_gcn.long(), reduction='none')
            loss_cls_tar_cnn = nl_criterion(tar_cnn_logits, pseudo_labels_cnn.long())
            loss_cls_tar_cnn = torch.mean(loss_cls_tar_cnn * confi_cnn, dim=0)
            loss_cls_tar_intra_gcn = nl_criterion(tar_gcn_intra_logits, pseudo_labels_intra_gcn.long())
            loss_cls_tar_intra_gcn = torch.mean(loss_cls_tar_intra_gcn * confi_intra_gcn, dim=0)
            loss_cls_tar = loss_cls_tar_cnn + loss_cls_tar_intra_gcn
        else:
            loss_cls_tar, loss_dist = torch.tensor(0.), torch.tensor(0.)

        loss_cls = loss_cls_src + args.coef_cls_tar * loss_cls_tar
        
        loss = loss_cls + 1e0 * loss_dist
        loss.backward()
        optimizer_GMCD.step()
        
        #@ Bank batch update
        with torch.no_grad():
            if len(src_cnn_logits_list) == 0 or len(src_cnn_logits_list) * src_cnn_logits_list[0].size(0) < 3000:
                [k.append(v) for k, v in zip([src_cnn_logits_list, src_cnn_feats_list, src_gcn_intra_logits_list, src_gcn_intra_feats_list, src_labels_list],
                    [src_cnn_logits, src_cnn_feats, src_gcn_intra_logits, src_gcn_intra_feats, label_src])]
            if not tar_stop and (len(tar_cnn_logits_list) == 0 or len(tar_cnn_logits_list) * tar_cnn_logits_list[0].size(0) < 3000):
                [k.append(v) for k, v in zip([tar_cnn_logits_list, tar_cnn_feats_list, tar_gcn_intra_logits_list, tar_gcn_intra_feats_list, tar_labels_list],
                    [tar_cnn_logits, tar_cnn_feats, tar_gcn_intra_logits, tar_gcn_intra_feats, label_tar])]
        
        Loss_Cls_CNN += loss_cls_cnn.item()
        Loss_Cls_GCN_Intra += loss_cls_gcn_intra.item()
        Loss_Cls_Tar += loss_cls_tar.item()
        Loss_Cls += loss_cls.item()
        Loss_Dist += loss_dist.item()
        
        pred_src_cnn = src_cnn_logits.data.max(1)[1]  # 如果是取[0]的话就是取坐标, [1]是取值
        pred_src_gcn_intra = src_gcn_intra_logits.data.max(1)[1]
        Correct_Src_CNN += pred_src_cnn.eq(label_src.data.view_as(pred_src_cnn)).cpu().sum()
        Correct_Src_GCN_Intra += pred_src_gcn_intra.eq(label_src.data.view_as(pred_src_gcn_intra)).cpu().sum()
        if pseudo_labels_cnn != None:
            Correct_PL_CNN += pseudo_labels_cnn.eq(label_tar.data.view_as(pseudo_labels_cnn)).cpu().sum()
            Correct_PL_Intra_GCN += pseudo_labels_intra_gcn.eq(label_tar.data.view_as(pseudo_labels_intra_gcn)).cpu().sum()
        len_tar_temp += tar_cnn_logits.shape[0]
        if len_tar_train_dataset - len_tar_temp >= 0:
            pred_tar_cnn = tar_cnn_logits.data.max(1)[1]
            pred_tar_gcn_intra = tar_gcn_intra_logits.max(1)[1]
            Correct_Tar_CNN += pred_tar_cnn.eq(label_tar.data.view_as(pred_tar_cnn)).cpu().sum()
            Correct_Tar_GCN_Intra += pred_tar_gcn_intra.eq(label_tar.data.view_as(pred_tar_gcn_intra)).cpu().sum()
            len_tar = len_tar_temp

        qbar.set_description(f"[Train Epoch {epoch+1}] cls_src: {loss_cls_cnn.item() + loss_cls_gcn_intra.item():.3f},"\
            f"cls_tar: {loss_cls_tar.item():.3f}, dist: {loss_dist.item():.3f}, sum: {loss.item():.3f}")

    #@ Bank datasets update
    with torch.no_grad():
        banks['src'] = {
            'cnn_probs': F.softmax(torch.cat(src_cnn_logits_list), dim=1),
            'cnn_feats': torch.cat(src_cnn_feats_list),
            'gcn_probs': F.softmax(torch.cat(src_gcn_intra_logits_list), dim=1),
            'gcn_feats': torch.cat(src_gcn_intra_feats_list),
            'gt_labels': torch.cat(src_labels_list)
        }
        banks['tar'] = {
            'cnn_probs': F.softmax(torch.cat(tar_cnn_logits_list), dim=1),
            'cnn_feats': torch.cat(tar_cnn_feats_list),
            'gcn_probs': F.softmax(torch.cat(tar_gcn_intra_logits_list), dim=1),
            'gcn_feats': torch.cat(tar_gcn_intra_feats_list),
            'gt_labels': torch.cat(tar_labels_list)
        }

    Acc_Src_CNN = Correct_Src_CNN.item() / len_src_dataset
    Acc_Src_GCN_Intra = Correct_Src_GCN_Intra.item() / len_src_dataset
    Acc_Tar_CNN = Correct_Tar_CNN.item() / len_tar
    Acc_Tar_GCN_Intra = Correct_Tar_GCN_Intra.item() / len_tar
    Acc_PL_CNN = Correct_PL_CNN.item() / len_tar
    Acc_PL_Intra_GCN = Correct_PL_Intra_GCN.item() / len_tar
    print(f"--> Train_Acc: {Acc_Src_CNN:.3f}, {Acc_Src_GCN_Intra:.3f}, {Acc_Tar_CNN:.3f}, {Acc_Tar_GCN_Intra:.3f}. PL: {Acc_PL_CNN:.3f}, {Acc_PL_Intra_GCN:.3f}")
    writer.add_scalars('Train_Loss_Cls', {'Cls_CNN': Loss_Cls_CNN/len_src_loader, 'Cls_GCN_Intra': Loss_Cls_GCN_Intra/len_src_loader,\
        'Cls_Tar': Loss_Cls_Tar/len_tar, 'Dist': Loss_Dist/len_tar}, epoch)
    writer.add_scalars('Train_Acc_Src', {'CNN': Acc_Src_CNN, 'GCN(Intra)': Acc_Src_GCN_Intra}, epoch)
    writer.add_scalars('Train_Acc_Tar', {'CNN': Acc_Tar_CNN, 'GCN(Intra)': Acc_Tar_GCN_Intra}, epoch)
    writer.add_scalars('Train_PL_Acc', {'CNN': Acc_PL_CNN, 'Intra_GCN': Acc_PL_Intra_GCN}, epoch)
    return model_GMCD
    

def test(model):
    model.eval()
    loss = 0
    correct = 0
    pred_list, label_list = [], []
    
    with torch.no_grad():
        print('Testing...')
        for _, data, label, _, _ in test_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            label = label - 1
            out = model(data)
            pred = out.data.max(1)[1]
            pred_list.append(pred.cpu().numpy())
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()

            label_list.append(label.cpu().numpy())
            loss += F.nll_loss(F.log_softmax(out, dim = 1), label.long()).item() # sum up batch loss

        loss /= len_tar_loader
        print('{} set: Average test loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n, | Test sample number: {:6}'.format(
            args.target_name, loss, correct, len_tar_dataset,
            100. * correct / len_tar_dataset, len_tar_dataset))
    return correct.item() / len_tar_dataset, correct, pred_list, label_list

if __name__ == '__main__':
    args = construct_argument()
    DEVICE = get_device(args.cuda)
    seed_worker(args.seed)
    dt = datetime.datetime.now()
    folder = dt.strftime('%Y-%m-%d-%H:%M:%S')
    args.output_path = os.path.join(args.output_path, folder)
    makeFolder(args)
    args.log_path = args.output_path
    args.save_path = args.output_path

    acc_test_list = [0. for i in range(args.num_trials)]
    acc_class_test_list = [{} for i in range(args.num_trials)]
    kappa_test_list = [0. for i in range(args.num_trials)]
    for flag in range(args.num_trials):
        #@ some hyperparameters depend on the dataset name
        if 'Houston' in args.data_path:
            args.source_name='Houston13'
            args.target_name='Houston18'
            args.lr = 1e-3
            args.coef_cls_tar = 1
        elif 'HyRANK' in args.data_path:
            args.source_name='Dioni'
            args.target_name='Loukia'
            args.lr = 1e-3
            args.coef_cls_tar = 1
        elif 'Pavia' in args.data_path:
            args.source_name='paviaU'
            args.target_name='paviaC'
            args.lr = 1e-3
            args.coef_cls_tar = 1
        elif 'S-H' in args.data_path:
            args.source_name='Hangzhou'
            args.target_name='Shanghai'
            args.lr = 1e-3
            args.coef_cls_tar = 1
            
        img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,
                                                                args.data_path)
        img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                                args.data_path)

        sample_num_src = len(np.nonzero(gt_src)[0]) # 统计的非零像素点的个数
        sample_num_tar = len(np.nonzero(gt_tar)[0])
        args.re_ratio = min(int(sample_num_tar/(sample_num_src * args.training_sample_ratio)), int(1/args.training_sample_ratio))
        # args.re_ratio = 1
        tmp = args.training_sample_ratio*args.re_ratio*sample_num_src/sample_num_tar
        training_sample_tar_ratio = tmp if tmp < 1 else 1
        print(f"SD Sample Rate = {args.training_sample_ratio:.2%}")
        print(f"TD Sample Rate = {training_sample_tar_ratio:.2%}")
        print(f"re_ratio = {args.re_ratio}")
        print(f"learning rate = {args.lr}")

        num_classes=gt_src.max()
        N_BANDS = img_src.shape[-1]
        hyperparams = vars(args)
        hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 
                            'device': DEVICE, 'center_pixel': False, 'supervision': 'full'})
        hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

        r = int(hyperparams['patch_size']/2) + 1
        img_src=np.pad(img_src,((r,r),(r,r),(0,0)),'symmetric') # raw: (W,H,C) => (W+2r, H+2r, C)
        img_tar=np.pad(img_tar,((r,r),(r,r),(0,0)),'symmetric') # 镜像填充, 所以不用使用特殊数值
        gt_src=np.pad(gt_src,((r,r),(r,r)),'constant',constant_values=(0,0))
        gt_tar=np.pad(gt_tar,((r,r),(r,r)),'constant',constant_values=(0,0))

        train_gt_src, _, training_set, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
        test_gt_tar, _, tesing_set, _ = sample_gt(gt_tar, 1, mode='random')
        train_gt_tar, _, _, _ = sample_gt(gt_tar, training_sample_tar_ratio, mode='random')
        img_src_con, img_tar_con, train_gt_src_con, train_gt_tar_con = img_src, img_tar, train_gt_src, train_gt_tar
        if tmp < 1:
            for i in range(args.re_ratio-1):
                img_src_con = np.concatenate((img_src_con,img_src))
                train_gt_src_con = np.concatenate((train_gt_src_con,train_gt_src))
                # img_tar_con = np.concatenate((img_tar_con,img_tar))
                # train_gt_tar_con = np.concatenate((train_gt_tar_con,train_gt_tar))
        args.classNum = int(gt_src.max())
        
        # Generate the dataset
        hyperparams_train = hyperparams.copy()

        train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
        g = torch.Generator() # 随机数生成类，训练和测试期间生成可重复的伪随机数字序列
        g.manual_seed(args.seed)
        train_loader = data.DataLoader(train_dataset,
                                        batch_size=hyperparams['batch_size'],
                                        pin_memory=True,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        shuffle=True)
        train_tar_dataset = HyperX(img_tar_con, train_gt_tar_con, **hyperparams)
        train_tar_loader = data.DataLoader(train_tar_dataset,
                                        pin_memory=True,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        batch_size=hyperparams['batch_size'],
                                        shuffle=True)
        test_dataset = HyperX(img_tar, test_gt_tar, flag='Test', **hyperparams)
        test_loader = data.DataLoader(test_dataset,
                                        pin_memory=True,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        batch_size=hyperparams['batch_size'])                      
        len_src_loader = len(train_loader) # batch_size为批次
        len_tar_train_loader = len(train_tar_loader)
        len_src_dataset = len(train_loader.dataset) # 总数量
        len_tar_train_dataset = len(train_tar_loader.dataset)
        len_tar_dataset = len(test_loader.dataset)
        len_tar_loader = len(test_loader)

        print(hyperparams)
        print("train samples :", len_src_dataset)
        print("train tar samples :", len_tar_train_dataset)

        correct, acc = 0, 0
        model_GMCD = GMCD.GraphMCD(img_src.shape[-1],num_classes=int(gt_src.max()), patch_size=hyperparams['patch_size']).to(DEVICE)
        if args.checkpoint != 'None':
            dic = torch.load(args.checkpoint)
            model_param_dict = dic['model_GMCD']
            model_dict = model_GMCD.state_dict()
            model_param_dict = {k: v for k, v in model_param_dict.items() if k in model_dict}
            model_dict.update(model_param_dict)
            model_GMCD.load_state_dict(model_dict)

        log_dir = os.path.join(args.log_path, args.source_name + '_lr_'+str(args.lr)+'_' + str(int(flag+1))+'times')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            shutil.copyfile('./tb.sh', os.path.join(log_dir, 'tb.sh'))
        writer = SummaryWriter(log_dir)

        banks = {'src': None, 'tar': None}
        cumulative_epochs = 0
        for epoch in range(args.num_epoch):
            if not args.isTest:
                model_GMCD = train(epoch, model_GMCD, args.num_epoch)
            if epoch % args.log_interval == 0:
                Test_Acc, t_correct, pred, label = test(model_GMCD)
                if t_correct > correct:
                    correct = t_correct
                    acc = Test_Acc
                    if acc > 0.60:
                        acc_test_list[flag] = acc
                        results = {}
                        metrics_hand_cal = metrics(np.concatenate(pred), np.concatenate(label), ignored_labels=hyperparams['ignored_labels'], n_classes=int(gt_src.max()))
                        class_acc_str = classification_report(np.concatenate(pred),np.concatenate(label),target_names=LABEL_VALUES_tar, digits=4)
                        class_acc_dict = classification_report(np.concatenate(pred),np.concatenate(label),target_names=LABEL_VALUES_tar, digits=4, output_dict=True)
                        results['class_acc_str'] = class_acc_str
                        results['class_acc_dict'] = class_acc_dict
                        results['metrics_hand_cal'] = metrics_hand_cal
                        acc_class_test_list[flag] = results
                        model_save_path = os.path.join(args.save_path, 'params_'+args.source_name+'_Acc'+str(int(acc*10000))+'_Kappa'+str(int(metrics_hand_cal['Kappa']*10000))+'.pkl')
                        kappa_test_list[flag] = metrics_hand_cal['Kappa']
                        print(class_acc_str)
                        print(f"kappa: {kappa_test_list[flag]:.2%}")
                        torch.save({'model_GMCD': model_GMCD.state_dict()}, model_save_path)
                        cumulative_epochs = 0
                else:
                    print(f"Best results have not been updated for {cumulative_epochs} epochs !")
                    cumulative_epochs += 1
                del pred, label
                gc.collect()

                print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
                    args.source_name, args.target_name, correct, 100. * correct / len_tar_dataset ))
            writer.add_scalars('Test_Acc', {'CNN_Test': Test_Acc}, epoch)
            
            if cumulative_epochs >= 150:
                break
        with open(os.path.join(args.save_path,'train_times_'+args.source_name+'.pickle'), 'wb') as f:
            pickle.dump({'acc_test_list': acc_test_list, 'kappa_test_list': kappa_test_list, 'acc_class_test_list': acc_class_test_list, 'lr':args.lr, 'class_num': args.classNum}, f)