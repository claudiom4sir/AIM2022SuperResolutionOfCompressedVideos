import torch
import os
import argparse
from net_stdf import MFVQE_SR_Y
from utils import save_data
from dataset import AIMChallengeTestset
from torch.utils.data import DataLoader
from tqdm import tqdm
import colors
import utils


def main(data_path, save_all):


    # path specification
    pretrained_path = 'best.pth'
    path_compressed_test = data_path
    output_dir = 'Results/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # define model
    device = torch.device('cuda:0')
    model = MFVQE_SR_Y()
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    print('> Loading pretrained model from ' + pretrained_path)
    model.load_state_dict(torch.load(pretrained_path)['model_state_dict'])

    # load testset
    testset = AIMChallengeTestset(path_compressed_test)
    testset_loader = DataLoader(testset, num_workers=8, prefetch_factor=8)

    print('Start testing the model...')
    model.eval()
    pbar = tqdm(total=len(testset_loader), ncols=100)
    with torch.no_grad():
        for batch, data in enumerate(testset_loader, 1):
            compressed_rgb, name = data
            if not save_all and int(name[0][4:-4]) % 10 != 0:
                pbar.update()
                continue
            compressed_rgb = compressed_rgb.to(device)

            compressed_ycbcr = torch.cat([colors.rgb2ycbcr(compressed_rgb[:, i:i + 3])
                                          for i in range(0, compressed_rgb.shape[1], 3)], dim=1)
            compressed_y = torch.cat([compressed_ycbcr[:, i:i + 1]
                                      for i in range(0, compressed_ycbcr.shape[1], 3)], dim=1)

            restored_y = model(compressed_y)

            compressed_ycbcr = compressed_ycbcr[:, 6:9]
            compressed_ycbcr = torch.nn.functional.interpolate(compressed_ycbcr, scale_factor=4,
                                                               mode='bicubic', align_corners=True)
            restored_ycbcr = torch.cat((restored_y, compressed_ycbcr[:, 1:]), dim=1)
            restored_rgb = colors.ycbcr2rgb(compressed_ycbcr)

            utils.save_data(restored_rgb, output_dir + name[0])
            pbar.update()
    pbar.close()


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='AIM 2022 Compressed Input Super-Resolution Challenge - Track 2 Video.'
                                                 'IVL team solution',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data parameters
    parser.add_argument("-dp", "--data_path", help="Path to dataset containing videos.",
                        default='../../../../dataset/val/', type=str)
    parser.add_argument("-sa", "--save_all", help="Save EACH restored frame as PNG? ",
                        default=False, type=bool)
    args = parser.parse_args()
    main(args.data_path, args.save_all)
