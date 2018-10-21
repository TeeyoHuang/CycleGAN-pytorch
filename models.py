import torch.nn as nn
import torch.nn.functional as F
import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


##############################
#           RESNET
##############################

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, in_channels, out_channels, res_blocks ):
        super(GeneratorResNet, self).__init__()
        #in_channels = args.input_nc
        #out_channels = args.output_nc
        #res_blocks = args.n_residual_blocks
        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(in_channels, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, out_channels, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################
class Discriminator_n_layers(nn.Module):
    def __init__(self,  n_D_layers, in_c):
        super(Discriminator_n_layers, self).__init__()

        n_layers = n_D_layers
        in_channels = in_c
        def discriminator_block(in_filters, out_filters, k=4, s=2, p=1, norm=True, sigmoid=False):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=k, stride=s, padding=p)]
            if norm:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if sigmoid:
                layers.append(nn.Sigmoid())
                print('use sigmoid')
            return layers

        sequence = [*discriminator_block(in_channels, 64, norm=False)] # (1,64,128,128)

        assert n_layers<=5

        if (n_layers == 1):
            'when n_layers==1, the patch_size is (16x16)'
            out_filters = 64* 2**(n_layers-1)

        elif (1 < n_layers & n_layers<= 4):
            '''
            when n_layers==2, the patch_size is (34x34)
            when n_layers==3, the patch_size is (70x70), this is the size used in the paper
            when n_layers==4, the patch_size is (142x142)
            '''
            for k in range(1,n_layers): # k=1,2,3
                sequence += [*discriminator_block(2**(5+k), 2**(6+k))]
            out_filters = 64* 2**(n_layers-1)

        elif (n_layers == 5):
            '''
            when n_layers==5, the patch_size is (286x286), lis larger than the img_size(256),
            so this is the whole img condition
            '''
            for k in range(1,4): # k=1,2,3
                sequence += [*discriminator_block(2**(5+k), 2**(6+k))]
            # k=4
            sequence += [*discriminator_block(2**9, 2**9)] #
            out_filters = 2**9

        num_of_filter = min(2*out_filters, 2**9)

        sequence += [*discriminator_block(out_filters, num_of_filter, k=4, s=1, p=1)]
        sequence += [*discriminator_block(num_of_filter, 1, k=4, s=1, p=1, norm=False, sigmoid=False)]

        self.model = nn.Sequential(*sequence)

    def forward(self, img_input ):
        return self.model(img_input)


####################################################
# Initialize generator and discriminator
####################################################
def Create_nets(args):
    generator_AB = GeneratorResNet(args.input_nc_A,   args.input_nc_B ,args.n_residual_blocks)
    discriminator_B = Discriminator_n_layers(args.n_D_layers, args.input_nc_B)
    generator_BA = GeneratorResNet(args.input_nc_B,   args.input_nc_A ,args.n_residual_blocks)
    discriminator_A = Discriminator_n_layers(args.n_D_layers, args.input_nc_A)

    if torch.cuda.is_available():
        generator_AB = generator_AB.cuda()
        discriminator_B = discriminator_B.cuda()
        generator_BA = generator_BA.cuda()
        discriminator_A = discriminator_A.cuda()

    if args.epoch_start != 0:
        # Load pretrained models
        generator_AB.load_state_dict(torch.load('saved_models/%s/G__AB_%d.pth' % (opt.dataset_name, opt.epoch)))
        discriminator_B.load_state_dict(torch.load('saved_models/%s/D__B_%d.pth' % (opt.dataset_name, opt.epoch)))
        generator_BA.load_state_dict(torch.load('saved_models/%s/G__BA_%d.pth' % (opt.dataset_name, opt.epoch)))
        discriminator_A.load_state_dict(torch.load('saved_models/%s/D__A_%d.pth' % (opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        generator_AB.apply(weights_init_normal)
        discriminator_B.apply(weights_init_normal)
        generator_BA.apply(weights_init_normal)
        discriminator_A.apply(weights_init_normal)

    return generator_AB, discriminator_B, generator_BA, discriminator_A
