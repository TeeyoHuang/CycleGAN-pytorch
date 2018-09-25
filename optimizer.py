import torch
import itertools
# Optimizers
def Get_optimizers(args, G_AB,G_BA, D__B, D__A):
    optimizer_G = torch.optim.Adam(
                    itertools.chain(G_AB.parameters(), G_BA.parameters()),
                    lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D_B = torch.optim.Adam(
                    D__B.parameters(),
                    lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D_A = torch.optim.Adam(
                        D__A.parameters(),
                        lr=args.lr, betas=(args.b1, args.b2))

    return optimizer_G, optimizer_D_B, optimizer_D_A

# Loss functions
def Get_loss_func(args):
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    if torch.cuda.is_available():
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()
    return criterion_GAN, criterion_cycle, criterion_identity
