import argparse
import os
import datetime
import torch
import numpy as np
import vae_model
from vae_model import Encoder, Decoder, VAE
from gmm import G_Cluster
from genotype_caller import GenotypeCaller

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):
    start_t = datetime.datetime.now()

    file_name = args.input
    data = np.loadtxt(file_name, dtype='float32', delimiter='\t')
    data_bk = data.copy()
    print("Data loaded from", file_name)
    print('dimensions of the data: ', data.shape)

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    setup_seed(args.seed)

    
    d1 = np.max([data.shape[1] // 5, 128])
    d2 = np.max([data.shape[1] // 10, 64])

    encoder = Encoder(data.shape[1], d1, d2, args.dimension)
    encoder = encoder.cuda()
    decoder = Decoder(args.dimension, d2, d1, data.shape[1])
    decoder = decoder.cuda()
    vae = VAE(encoder, decoder, args.dimension)
    vae = vae.cuda()

    optimizer = torch.optim.RMSprop(vae.parameters(), lr=args.lr)

    vae.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for step, x in vae_model.xs_gen(data, args.batch_size, 1):
            x = torch.from_numpy(x)
            x = x.cuda()
            tv_m = x == 3
            x[tv_m] = 0

            p = vae(x)
            ce = vae_model.ce_loss(p, x, torch.logical_not(tv_m))
            kl = vae_model.kl_loss(vae.z_mean, vae.z_sigma)
            loss = 0.0001 * kl + ce
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if step % 50 == 0:
            #     print("ce", ce)
            #     print("kl", kl)
            #     print("step:", step, "| train loss: %.4f" % loss.data.cpu().numpy())

        print("epoch:", epoch, "| train loss: %.4f" % total_loss.data.cpu().numpy())

    a = []
    data = data_bk.copy()
    data[data == 3] = 0
    vae.eval()
    for step, x in vae_model.xs_gen(data, args.batch_size, 0):
        x = torch.from_numpy(x)
        x = x.cuda()
        with torch.no_grad():
            vae(x)
            mu = vae.z_mean.cpu().detach().numpy()
            a.append(mu)

    for id, mu in enumerate(a):
        if id == 0:
            features = mu
        else:
            features = np.r_[features, mu]

    print('clustering the cells...')
    if args.Kmax <= 0:
        kmax = np.max([1, features.shape[0] // 10])
    else:
        kmax = np.min([args.Kmax, features.shape[0] // 10])
    label_p, K = G_Cluster(features, kmax).cluster()
    print('inferred number of clusters: {}'.format(K))

    data = data_bk.copy()
    print('estimating genotypes...')
    genotypes, alpha, beta = GenotypeCaller(label_p, data).estimate_genotypes()
    print('estimated alpha: ', alpha, ', beta: ', beta)

    output_dir = args.output
    file_o = open(output_dir + '/para.txt', 'w')
    seq = ['{:.5f}'.format(alpha), '\t', '{:.5f}'.format(beta), '\n']
    file_o.writelines(seq)
    file_o.close()

    file_o = open(output_dir + '/labels.txt', 'w')
    np.savetxt(file_o, np.reshape(label_p, (1, data_bk.shape[0])), fmt='%d', delimiter=',')
    file_o.close()

    file_o = open(output_dir + '/clusters.txt', 'w')
    np.savetxt(file_o, genotypes, fmt='%d', delimiter=',')
    file_o.close()

    end_t = datetime.datetime.now()
    print('elapsed time: ', (end_t-start_t).seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="bmVAE")
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--Kmax', type=int, default=0)
    parser.add_argument('--dimension', type=int, default=3)
    parser.add_argument('--input', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
