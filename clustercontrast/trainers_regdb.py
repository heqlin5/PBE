from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.nn import functional as F


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx 
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class PGLR(nn.Module):
    """ Part-guided label refinement """
    def __init__(self, lam=0.5):
        super(PGLR, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.lam = lam

    def forward(self, logits_g, logits_p, targets, ca):
        targets = torch.zeros_like(logits_g).scatter_(1, targets.unsqueeze(1), 1)
        w = torch.softmax(ca, dim=1)  # B * P
        w = torch.unsqueeze(w, 1)  # B * 1 * P
        # print(w)
        preds_p = self.softmax(logits_p)  # B * C * P
        ensembled_preds = (preds_p * w).sum(2).detach()  # B * class_num
        refined_targets = self.lam * targets + (1-self.lam) * ensembled_preds

        log_preds_g = self.logsoftmax(logits_g)
        loss = (-refined_targets * log_preds_g).sum(1).mean()
        return loss

class AALS(nn.Module):
    """ Agreement-aware label smoothing """
    def __init__(self):
        super(AALS, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, logits, targets, ca):
        log_preds = self.logsoftmax(logits)  # B * C
        targets = torch.zeros_like(log_preds).scatter_(1, targets.unsqueeze(1), 1)
        uni = (torch.ones_like(log_preds) / log_preds.size(-1)).cuda()

        loss_ce = (- targets * log_preds).sum(1)
        loss_kld = F.kl_div(log_preds, uni, reduction='none').sum(1)
        loss = (ca * loss_ce + (1-ca) * loss_kld).mean()
        return loss

class ClusterContrastTrainer_pretrain_joint(object):
    def __init__(self, encoder, num_part=3, memory=None, memory_part = None):
        super(ClusterContrastTrainer_pretrain_joint, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_ir_part = memory_part
        self.memory_rgb = memory
        self.memory_rgb_part = memory_part
        self.score_rgb = None
        self.score_ir = None
        self.num_part = num_part
        # self.lam = 0.5
        self.criterion_pglr = PGLR(lam=0.9).cuda()
        self.criterion_aals = AALS().cuda()

    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer, print_freq=10, train_iters=400,score_rgb=None,score_ir=None):
        self.encoder.train()
        self.score_rgb = score_rgb
        self.score_ir = score_ir

        # print(self.score_rgb.shape)
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir,labels_ir, indexes_ir,ca_ir = self._parse_data_ir(inputs_ir) #inputs_ir1
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,ca_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            ca_rgb = torch.cat((ca_rgb,ca_rgb),0)


            _,f_out_rgb, f_out_ir, f_out_rgb_part, f_out_ir_part, labels_rgb,labels_ir,pool_rgb,pool_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)


            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)


            loss = loss_ir+loss_rgb# + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, _, indexes = inputs
        ca = self.score_rgb[indexes]
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),ca.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        ca = self.score_ir[indexes]
        return imgs.cuda(), pids.cuda(), indexes.cuda(),ca.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)



class ClusterContrastTrainer_pretrain_camera_interC(object):
    def __init__(self, encoder, num_part=3, memory=None, memory_part = None):
        super(ClusterContrastTrainer_pretrain_camera_interC, self).__init__()

        self.encoder = encoder
        self.memory_ir = memory
        self.memory_ir_part = memory_part
        self.memory_rgb = memory
        self.memory_rgb_part = memory_part
        self.score_rgb = None
        self.score_ir = None
        self.num_part = num_part
        # self.lam = 0.5
        self.criterion_pglr = PGLR(lam=0.9).cuda()
        self.criterion_aals = AALS().cuda()


    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
        all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
                 print_freq=10, train_iters=400,score_rgb=None,score_ir=None):
        self.encoder.train()
        self.score_rgb = score_rgb
        self.score_ir = score_ir

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        
        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir,labels_ir, indexes_ir,cids_ir, ca_ir = self._parse_data_ir(inputs_ir) #inputs_ir1
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb, ca_rgb = self._parse_data_rgb(inputs_rgb)

            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)
            ca_rgb = torch.cat((ca_rgb, ca_rgb), 0)
            _,f_out_rgb,f_out_ir,f_out_rgb_part,f_out_ir_part, labels_rgb,labels_ir,cids_rgb,cids_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,cid_rgb=cids_rgb,cid_ir=cids_ir)
            

            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss_camera_ir = torch.tensor([0.]).cuda()
            loss_camera_rgb = torch.tensor([0.]).cuda()


################camera

            
##################
            lamda_c = 0.1


            ###regdb
            loss = loss_ir+loss_rgb# + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, cids, indexes = inputs
        ca = self.score_rgb[indexes]
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),ca.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, cids, indexes = inputs
        ca = self.score_ir[indexes]
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),ca.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,cid_rgb=None,cid_ir=None):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir)


class ClusterContrastTrainer_pretrain_camera_interM(object):
    def __init__(self, encoder, num_part=3, memory=None, memory_part = None):
        super(ClusterContrastTrainer_pretrain_camera_interM, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_ir_part = memory_part
        self.memory_rgb = memory
        self.memory_rgb_part = memory_part
        self.score_rgb = None
        self.score_ir = None
        self.num_part = num_part
        # self.lam = 0.5
        self.criterion_pglr = PGLR(lam=0.9).cuda()
        self.criterion_aals = AALS().cuda()


    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
        all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
                 print_freq=10, train_iters=400,score_rgb=None,score_ir=None):
        self.encoder.train()
        self.score_rgb = score_rgb
        self.score_ir = score_ir

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()


        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            inputs_ir,labels_ir, indexes_ir,cids_ir,ca_ir = self._parse_data_ir(inputs_ir) #inputs_ir1
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,ca_rgb = self._parse_data_rgb(inputs_rgb)

            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)

            cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)
            ca_rgb = torch.cat((ca_rgb, ca_rgb), 0)

            _,f_out_rgb,f_out_ir,f_out_rgb_part,f_out_ir_part,labels_rgb,labels_ir,cids_rgb,cids_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,cid_rgb=cids_rgb,cid_ir=cids_ir)

            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)

            
            loss_contrast_rgb = 0.
            loss_contrast_ir = 0.
            if self.num_part > 0:
                # if epoch >= self.aals_epoch:
                # if epoch >= 5:
                if epoch >= 0:
                    for part in range(self.num_part):
                        loss_contrast_rgb += self.memory_rgb_part[part](f_out_rgb_part[:, :, part], labels_rgb)
                        loss_contrast_ir += self.memory_ir_part[part](f_out_ir_part[:, :, part], labels_ir)


                loss_contrast_rgb /= self.num_part
                loss_contrast_ir /= self.num_part

           
            loss = loss_ir+loss_rgb + loss_contrast_rgb + loss_contrast_ir# + loss_tri
   

            # loss = loss_ir+loss_rgb

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'Loss ir part {:.3f}\t'
                      'Loss rgb part {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb,loss_contrast_ir,loss_contrast_rgb))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, cids, indexes = inputs
        ca = self.score_rgb[indexes]
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),ca.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, cids, indexes = inputs
        ca = self.score_ir[indexes]
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),ca.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,cid_rgb=None,cid_ir=None):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir)








