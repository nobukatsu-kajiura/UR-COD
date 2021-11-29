import torch
import torch.nn.functional as F
from torch.autograd import Variable

import os
import argparse
from datetime import datetime
from model.URCOD import Generator
from utils.loss import smoothness_loss, structure_loss, flooding_dice_bce_loss
from utils.dataloader import get_loader
from utils.utils import adjust_lr, linear_annealing, visualize

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--name', type=str, default='UR-SINetv2')
parser.add_argument('--dataset', type=str, default='SINetv2')
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
parser.add_argument('--beta1_gen', type=float, default=0.5,help='beta of Adam for generator')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
parser.add_argument('--latent_dim', type=int, default=3, help='latent dim')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of camouflaged feat')
parser.add_argument('--sm_weight', type=float, default=0.1, help='weight for smoothness loss')
parser.add_argument('--edge_weight', type=float, default=10.0, help='weight for edge loss')
parser.add_argument('--lat_weight', type=float, default=10.0, help='weight for latent loss')
parser.add_argument('--vae_loss_weight', type=float, default=0.4, help='weight for vae loss')
parser.add_argument('--mse_loss_weight', type=float, default=0.1, help='weight for mse loss')
opt = parser.parse_args()

torch.cuda.set_device(opt.gpu)
generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)
generator.cuda()
generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen, betas=[opt.beta1_gen, 0.999], weight_decay=opt.weight_decay)

image_root = './data/TrainDataset/Img_RGB/'
gtmask_root = './data/TrainDataset/GT_Mask/'
gtedge_root = './data/TrainDataset/GT_Edge/'
pseudomask_root = './data/TrainDataset/Pseudo_Mask/{}/'.format(opt.dataset)
gray_root = './data/TrainDataset/Img_Gray/'

train_loader, training_set_size = get_loader(image_root, gtmask_root, gtedge_root, pseudomask_root, gray_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)
train_z = torch.FloatTensor(training_set_size, opt.latent_dim).normal_(0, 1).cuda()

mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
smooth_loss = smoothness_loss(size_average=True)
dice_bce_loss = flooding_dice_bce_loss()

for epoch in range(1, opt.epoch+1):
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))

    for i, pack in enumerate(train_loader, start=1):
        images, gtmasks, gtmask_rgbs, gtedges, pseudomasks, grays, index_batch = pack
        images = Variable(images).cuda()
        gtmasks = Variable(gtmasks).cuda()
        gtmask_rgbs = Variable(gtmask_rgbs).cuda()
        gtedges = Variable(gtedges).cuda()
        pseudomasks = Variable(pseudomasks).cuda()
        grays = Variable(grays).cuda()

        gt_pseudos = torch.cat((gtmask_rgbs, gtedges), 1)
        pred_post, pred_prior, latent_loss, pseudo_pred_post, pseudo_pred_prior, pseudoedges = generator.forward(images, pseudomasks, gtmasks)

        # gen_loss_cvae
        smoothLoss_post = opt.sm_weight * smooth_loss(torch.sigmoid(pred_post), gtmasks)
        mse_loss_post = opt.mse_loss_weight * mse_loss(torch.sigmoid(pseudo_pred_post), gt_pseudos)
        ref_loss = structure_loss(pred_post, gtmasks) + smoothLoss_post + mse_loss_post
        anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
        latent_loss = opt.lat_weight * anneal_reg * latent_loss
        gen_loss_cvae = ref_loss + latent_loss
        gen_loss_cvae = opt.vae_loss_weight * gen_loss_cvae

        # gen_loss_gsnn
        smoothLoss_prior = opt.sm_weight * smooth_loss(torch.sigmoid(pred_prior), gtmasks)
        mse_loss_prior = opt.mse_loss_weight * mse_loss(torch.sigmoid(pseudo_pred_prior), gt_pseudos)
        gen_loss_gsnn = structure_loss(pred_prior, gtmasks) + smoothLoss_prior + mse_loss_prior
        gen_loss_gsnn = (1-opt.vae_loss_weight) * gen_loss_gsnn

        edge_loss = dice_bce_loss(gtedges, pseudoedges) * opt.edge_weight
        gen_loss = gen_loss_cvae + gen_loss_gsnn + edge_loss

        generator_optimizer.zero_grad()
        gen_loss.backward()
        generator_optimizer.step()

        prior_mask, prior_edge = torch.split(pseudo_pred_prior, 3, 1)
        visualize(pseudoedges, 'pseudo_edge', opt)
        visualize(torch.sigmoid(pred_post), 'post_init', opt)
        visualize(torch.sigmoid(pred_prior), 'prior_init', opt)
        visualize(torch.sigmoid(prior_edge), 'prior_edge', opt)

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen vae Loss: {:.4f}, gen gsnn Loss: {:.4f}, edge Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, gen_loss_cvae.data, gen_loss_gsnn.data, edge_loss.data))

    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

    save_path = 'checkpoints/{}/'.format(opt.name)
    os.makedirs(save_path, exist_ok=True)
    if epoch % 10 == 0:
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
