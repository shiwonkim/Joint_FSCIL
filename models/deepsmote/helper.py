from tqdm import tqdm
from utils.tools import *
from sklearn.neighbors import NearestNeighbors

import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


def biased_get_class1(c, dec_x, dec_y):
    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]

    return xbeg, ybeg


def G_SM1(X, y, n_to_sample, cl):
    # fitting the model
    n_to_sample = n_to_sample.item()
    n_neigh = 3
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    # generating samples
    base_indices = np.random.choice(list(range(len(X))), n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    samples = X_base + np.multiply(np.random.rand(n_to_sample, 1),
                                   X_neighbor - X_base)

    # use 10 as label because 0 to 9 real classes and 1 fake/smoted = 10
    return samples, [cl] * n_to_sample


def session_generate(encoder, decoder, dec_x, dec_y, save_path='./'):
    encoder.eval()
    decoder.eval()

    resx = []
    resy = []
    class_list, imbal = dec_y.unique(return_counts=True)
    imbal_max = max(imbal)
    target_classes = class_list[torch.where(imbal < imbal_max)[0]]

    for i in target_classes:
        xclass, yclass = biased_get_class1(i, dec_x, dec_y)

        # encode xclass to feature space
        xclass = torch.Tensor(xclass)
        xclass = xclass.cuda()
        xclass = encoder(xclass)

        xclass = xclass.detach().cpu().numpy()
        n = imbal_max - imbal[i]
        xsamp, ysamp = G_SM1(xclass, yclass, n, i)
        ysamp = np.array(ysamp)

        """to generate samples for resnet"""
        xsamp = torch.Tensor(xsamp)
        xsamp = xsamp.cuda()
        ximg = decoder(xsamp)

        ximn = ximg.detach().cpu().numpy()
        resx.append(ximn)
        resy.append(ysamp)

    resx1 = np.vstack(resx)
    resy1 = np.hstack(resy)

    np.save(os.path.join(save_path, 'images.npy'), resx1)
    np.save(os.path.join(save_path, 'labels.npy'), resy1)


def extract_from_concatdataset(concat_dataset):
    dec_x = []
    dec_y = []

    for dataset in concat_dataset.datasets:
        for x, y in dataset:
            dec_x.append(x)
            dec_y.append(y)

    dec_x = torch.stack(dec_x)
    dec_y = torch.tensor(dec_y)

    return dec_x, dec_y


def session_train(encoder, decoder, trainloader,
                  enc_optim, dec_optim, criterion, dec_x, dec_y):
    train_loss = []
    tmse_loss = []
    tdiscr_loss = []

    encoder.train()
    decoder.train()

    tqdm_gen = tqdm(trainloader)

    torch.cuda.empty_cache()

    for i, batch in enumerate(tqdm_gen, 1):
        encoder.zero_grad()
        decoder.zero_grad()

        images, train_label = [_.cuda() for _ in batch]

        z_hat = encoder(images)

        x_hat = decoder(z_hat)  # decoder outputs tanh
        # print('xhat ', x_hat.size())
        # print(x_hat)
        mse = criterion(x_hat, images)
        # print('mse ',mse)

        tc = np.random.choice(dec_y.max().item(), 1)
        # tc = 9
        xbeg = dec_x[dec_y == tc[0]]
        ybeg = dec_y[dec_y == tc[0]]
        xlen = len(xbeg)
        nsamp = min(xlen, 100)
        ind = np.random.choice(list(range(len(xbeg))), nsamp, replace=False)
        xclass = xbeg[ind]
        yclass = ybeg[ind]

        xclen = len(xclass)
        # print('xclen ',xclen)
        xcminus = np.arange(1, xclen)
        # print('minus ',xcminus.shape,xcminus)

        xcplus = np.append(xcminus, 0)
        # print('xcplus ',xcplus)
        xcnew = (xclass[[xcplus], :])
        # xcnew = np.squeeze(xcnew)
        xcnew = xcnew.reshape(xcnew.shape[1], xcnew.shape[2], xcnew.shape[3], xcnew.shape[4])
        # print('xcnew ',xcnew.shape)

        xcnew = torch.Tensor(xcnew)
        xcnew = xcnew.cuda()

        # encode xclass to feature space
        xclass = torch.Tensor(xclass)
        xclass = xclass.cuda()
        xclass = encoder(xclass)
        # print('xclass ',xclass.shape)

        xclass = xclass.detach().cpu().numpy()

        xc_enc = (xclass[[xcplus], :])
        xc_enc = np.squeeze(xc_enc)
        # print('xc enc ',xc_enc.shape)

        xc_enc = torch.Tensor(xc_enc)
        xc_enc = xc_enc.cuda()

        ximg = decoder(xc_enc)

        mse2 = criterion(ximg, xcnew)

        comb_loss = mse2 + mse
        comb_loss.backward()

        enc_optim.step()
        dec_optim.step()

        with torch.no_grad():
            train_loss.append(comb_loss.item())
            tmse_loss.append(mse.item())
            tdiscr_loss.append(mse2.item())

    train_loss = sum(train_loss) / (len(tqdm_gen)+1)
    tmse_loss = sum(tmse_loss) / (len(tqdm_gen)+1)
    tdiscr_loss = sum(tdiscr_loss) / (len(tqdm_gen)+1)

    return train_loss, tmse_loss, tdiscr_loss
