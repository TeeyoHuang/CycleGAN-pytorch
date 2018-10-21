import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from options import TrainOptions
from models import Create_nets
from datasets import Get_dataloader

from optimizer import Get_loss_func, Get_optimizers
from utils import ReplayBuffer, LambdaLR, sample_images

#load the args
args = TrainOptions().parse()
# Calculate output of size discriminator (PatchGAN)
patch = (1, args.img_height//(2**args.n_D_layers) - 2 , args.img_width//(2**args.n_D_layers) - 2)

# Initialize generator and discriminator
G__AB, D__B, G__BA, D__A = Create_nets(args)

# Loss functions
criterion_GAN, criterion_cycle, criterion_identity = Get_loss_func(args)
# Optimizers
optimizer_G, optimizer_D_B, optimizer_D_A = Get_optimizers(args, G__AB, G__BA, D__B, D__A )
# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.epoch_num, args.epoch_start, args.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(args.epoch_num, args.epoch_start, args.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(args.epoch_num, args.epoch_start, args.decay_epoch).step)

# Configure dataloaders
train_dataloader,test_dataloader,_ = Get_dataloader(args)

# Buffers of previously generated samples
fake_Y_A_buffer = ReplayBuffer()
fake_X_B_buffer = ReplayBuffer()


# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(args.epoch_start, args.epoch_num):
    for i, batch in enumerate(train_dataloader):


        ###############################################################################
        #### You can regard the A and B as two defferent styles;
        #### X and Y as two defferent images which in two defferent styles respectively
        #### So the generator_AB change the style from A to B; generator_BA change the style from B to A
        ################################################################################

        # Set model input
        real_X_A = Variable(batch['X'].type(torch.FloatTensor).cuda())
        real_Y_B = Variable(batch['Y'].type(torch.FloatTensor).cuda())

         # Adversarial ground truths
        valid = Variable(torch.FloatTensor(np.ones((real_X_A.size(0), *patch))).cuda(), requires_grad=False)
        fake = Variable(torch.FloatTensor(np.zeros((real_X_A.size(0), *patch))).cuda(), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()
        # Identity loss
        loss_id_A = criterion_identity(G__BA(real_X_A), real_X_A)
        loss_id_B = criterion_identity(G__AB(real_Y_B), real_Y_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_X_B = G__AB(real_X_A)
        pred_fake = D__B(fake_X_B)
        #print(pred_fake.shape,valid.shape)
        loss_GAN_AB = criterion_GAN(pred_fake, valid)

        fake_Y_A = G__BA(real_Y_B)
        pred_fake = D__A(fake_Y_A)
        loss_GAN_BA = criterion_GAN(pred_fake, valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_X_A = G__BA(fake_X_B)
        loss_cycle_A = criterion_cycle(recov_X_A, real_X_A)
        recov_Y_B = G__AB(fake_Y_A)
        loss_cycle_B = criterion_cycle(recov_Y_B, real_Y_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G =    loss_GAN + \
                    args.lambda_cyc * loss_cycle + \
                    args.lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()

         # -----------------------
        #  Train Discriminator A
        # -----------------------
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = D__A(real_X_A)
        loss_real = criterion_GAN(pred_real, valid)
        # Fake loss (on batch of previously generated samples)
        fake_Y_A_ = fake_Y_A_buffer.push_and_pop(fake_Y_A)
        pred_fake = D__A(fake_Y_A_.detach())
        loss_fake = criterion_GAN(pred_fake, fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = D__B(real_Y_B)
        loss_real = criterion_GAN(pred_real, valid)
        # Fake loss (on batch of previously generated samples)
        fake_X_B_ = fake_X_B_buffer.push_and_pop(fake_X_B)
        pred_fake = D__B(fake_X_B_.detach())
        loss_fake = criterion_GAN(pred_fake, fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2


        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(train_dataloader) + i
        batches_left = args.epoch_num * len(train_dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s" %
                                                        (epoch+1, args.epoch_num,
                                                        i, len(train_dataloader),
                                                        loss_D.data.cpu(), loss_G.data.cpu(),
                                                        loss_GAN.data.cpu(), loss_cycle.data.cpu(),
                                                        loss_identity.data.cpu(), time_left))

        # If at sample interval save image
        if batches_done % args.sample_interval == 0:
            sample_images(args,G__AB,G__BA, test_dataloader, epoch, batches_done)




    # Update learning rates
    lr_scheduler_G.step(epoch)
    lr_scheduler_D_B.step(epoch)
    lr_scheduler_D_A.step(epoch)

    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G__AB.state_dict(), '%s/%s/G__AB_%d.pth' % (args.model_result_dir, args.dataset_name, epoch))
        torch.save(G__BA.state_dict(), '%s/%s/G__BA_%d.pth' % (args.model_result_dir, args.dataset_name, epoch))
        torch.save(D_A.state_dict(), '%s/%s/D__A_%d.pth' % (args.model_result_dir, args.dataset_name, epoch))
        torch.save(D_B.state_dict(), '%s/%s/D__B_%d.pth' % (args.model_result_dir, args.dataset_name, epoch))
