from __future__ import print_function, absolute_import

import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .metrics import compute_distance_matrix
from .utils.meters import AverageMeter


class Trainer_UDA(object):
    def __init__(self, encoder, memory, source_classes):
        super(Trainer_UDA, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.source_classes = source_classes

    def train(self, epoch, data_loader_source, data_loader_target,
              optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_s = AverageMeter()
        losses_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            s_inputs, s_targets, _ = self._parse_data(source_inputs)
            t_inputs, _, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            device_num = torch.cuda.device_count()
            B, C, H, W = s_inputs.size()

            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)

            s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # forward
            f_out = self._forward(inputs)

            # de-arrange batch
            f_out = f_out.view(device_num, -1, f_out.size(-1))
            f_out_s, f_out_t = f_out.split(f_out.size(1) // 2, dim=1)
            f_out_s, f_out_t = f_out_s.contiguous().view(-1, f_out.size(-1)), f_out_t.contiguous().view(-1,
                                                                                                        f_out.size(-1))

            # compute loss with the hybrid memory
            loss_s = self.memory(f_out_s, s_targets)
            loss_t = self.memory(f_out_t, t_indexes + self.source_classes)

            loss = loss_s + loss_t
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_s.update(loss_s.item())
            losses_t.update(loss_t.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_s {:.3f} ({:.3f})\t'
                      'Loss_t {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_s.val, losses_s.avg,
                              losses_t.val, losses_t.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


class Trainer_USL(object):
    def __init__(self, encoder, memory):
        super(Trainer_USL, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, _, indexes = self._parse_data(inputs)

            # forward
            f_out = self._forward(inputs)

            # compute loss with the instance memory
            loss = self.memory(f_out, indexes)

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
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)

            # forward
            f_out = self._forward(inputs)

            # compute loss with the cluster memory
            loss = self.memory(f_out, labels)

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
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


class Trainer_AL(object):
    def __init__(self, encoder, memory=None, margin=1.0, lmda=1.0):
        super(Trainer_AL, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.margin = margin
        self.lmda = lmda
        self.margin_loss = nn.MarginRankingLoss(margin=self.margin).cuda()

    def train(self, epoch, data_loader, optimizer, al_masks, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        losses_cont = AverageMeter()
        losses_triplet = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, _, indexes = self._parse_data(inputs)  # inputs.shape: batch_size * channel * H * W

            # forward
            f_out = self._forward(inputs)

            # compute unsupervised contrastive loss with the instance memory
            loss_cont = self.memory(f_out, indexes)

            # compute supervised triplet loss
            B = inputs.size(0)
            dist = compute_distance_matrix(f_out, self.memory.features.clone())
            dist_ap, dist_an = [], []
            for j in range(B):
                k = indexes[j]
                if (al_masks[k] == 1).any() and (al_masks[k] == -1).any():
                    dist_ap.append(dist[j][al_masks[k] == 1].max().unsqueeze(0))
                    dist_an.append(dist[j][al_masks[k] == -1].min().unsqueeze(0))
            if len(dist_ap) == 0 and len(dist_an) == 0:
                loss_triplet = torch.tensor(0.).cuda()
            else:
                dist_ap = torch.cat(dist_ap)
                dist_an = torch.cat(dist_an)
                y = torch.ones_like(dist_an)
                loss_triplet = self.margin_loss(dist_an, dist_ap, y)

            loss = loss_cont + self.lmda * loss_triplet

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            losses_cont.update(loss_cont.item())
            losses_triplet.update(loss_triplet.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss_cont {:.3f} ({:.3f})\t'
                      'Loss_triplet {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_cont.val, losses_cont.avg,
                              losses_triplet.val, losses_triplet.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

