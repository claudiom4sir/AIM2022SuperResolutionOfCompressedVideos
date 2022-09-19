import torchvision.transforms.functional
from torch.utils.data.dataset import Dataset as D
import os
import numpy as np
import random
import torch
import torchvision as tv
import cv2
import sys
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from tqdm import tqdm
import colors


def generate_indices(end, temp_size=5, temp_stride=1):
    indices = []
    indices.append(np.array([0, 0, 0, 1, 2]))
    indices.append(np.array([0, 0, 1, 2, 3]))
    for i in range(0, end - temp_size + 1, temp_stride):
        ind = np.arange(i, temp_size + i)
        indices.append(ind)
    indices.append(np.array([end-4, end-3, end-2, end-1, end-1]))
    indices.append(np.array([end-3, end-2, end-1, end-1, end-1]))
    return indices


class AIMChallengeTestset(D):

    def __init__(self, path_compressed_data, path_gt_data=None, temp_size=5, temp_stride=1,
                 input_h=128, input_w=240):
        super(AIMChallengeTestset, self).__init__()
        self.path_compressed_data = path_compressed_data
        self.path_gt_data = path_gt_data
        self.h = input_h
        self.w = input_w
        self.is_challenge_test = path_gt_data is None
        self.temp_size = temp_size
        comp_videos_folders = os.listdir(self.path_compressed_data)
        gt_videos_folders = [] if self.is_challenge_test else os.listdir(self.path_gt_data)
        # length check
        videos = list(set(comp_videos_folders + gt_videos_folders))
        if not self.is_challenge_test:
            if len(videos) != len(comp_videos_folders) or len(videos) != len(gt_videos_folders):
                print('> Error: compressed and GT data do not match')
                sys.exit(1)
            for video in videos:
                if len(os.listdir(path_compressed_data + video)) != len(os.listdir(path_gt_data + video)):
                    print('> Error: compressed and GT data do not match in folder ' + video)
                    sys.exit(1)
        self.data = {}  # key -> video name, elements = list of gt and compressed frames
        self.indices = []
        print('> Loading video sequences...')
        # iterate through gt videos
        videos.sort()
        for video in videos:
            compressed_path = path_compressed_data + video
            if not self.is_challenge_test:
                gt_path = path_gt_data + video
                if len(os.listdir(compressed_path)) != len(os.listdir(gt_path)):
                    print('> Error: video length mismatch for video' + video)
                    sys.exit(-1)
            frames = os.listdir(compressed_path)
            frames.sort()
            for frame in frames:
                if not self.is_challenge_test and not os.path.isfile(gt_path + '/' + frame):
                    print('> Error: cannot find file ' + gt_path + '/' + frame)
                    sys.exit(-1)
                if int(video) not in self.data.keys():
                    self.data[int(video)] = []
                self.data[int(video)].append(video + '/' + frame)
            for indices in generate_indices(len(self.data[int(video)]), temp_size, temp_stride):
                self.indices.append([int(video), indices])
        print('> Done.')

    def __getitem__(self, index):
        name, indices = self.indices[index]
        frame_names = self.data[name]
        target_name = frame_names[indices[self.temp_size//2]]

        path_compressed_frames = []
        for index in indices:
            path_compressed_frames.append(self.path_compressed_data + frame_names[index])

        compressed_frames = []
        for frame_path in path_compressed_frames:
            compressed = torch.from_numpy(np.float32(cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB))
                                          / 255).permute(2, 0, 1)
            compressed_frames.append(compressed)
        compressed_frames = torch.cat([frame for frame in compressed_frames], dim=0)

        return compressed_frames, target_name

    def __len__(self):
        return len(self.indices)


class Dataset(D):

    def __init__(self, path_compressed_data, path_gt_data, input_patch_size=32, temp_size=5, temp_stride=1,
                 is_test=False, input_h=128, input_w=240, max_video_number=1e4, max_seq_length=1e4,
                 initial_seq_number=1, super_resolution=True):
        super(Dataset, self).__init__()
        self.path_compressed_data = path_compressed_data
        self.path_gt_data = path_gt_data
        self.input_patch_size = input_patch_size
        self.is_test = is_test
        self.h = input_h
        self.w = input_w
        self.temp_size = temp_size
        self.sr = super_resolution
        comp_videos_folders = os.listdir(self.path_compressed_data)
        gt_videos_folders = os.listdir(self.path_gt_data)
        # length check
        videos = list(set(comp_videos_folders + gt_videos_folders))
        if len(videos) != len(comp_videos_folders) or len(videos) != len(gt_videos_folders):
            print('> Error: compressed and GT data do not match')
            sys.exit(1)
        for video in videos:
            if len(os.listdir(path_compressed_data + video)) != len(os.listdir(path_gt_data + video)):
                print('> Error: compressed and GT data do not match in folder ' + video)
                sys.exit(1)
        self.data = {}  # key -> video name, elements = list of gt and compressed frames
        self.indices = []
        print('> Loading video sequences...')
        # iterate through gt videos
        videos.sort()
        videos = videos[initial_seq_number - 1:]
        for video_number, video in enumerate(videos, 1):
            compressed_path = path_compressed_data + video
            gt_path = path_gt_data + video
            if len(os.listdir(compressed_path)) != len(os.listdir(gt_path)):
                print('> Error: video length mismatch for video' + video)
                sys.exit(-1)
            frames = os.listdir(compressed_path)
            frames.sort()
            for frame_number, frame in enumerate(frames, 1):
                if not os.path.isfile(gt_path + '/' + frame):
                    print('> Error: cannot find file ' + gt_path + '/' + frame)
                    sys.exit(-1)
                if int(video) not in self.data.keys():
                    self.data[int(video)] = []
                self.data[int(video)].append(video + '/' + frame)
                if frame_number >= max_seq_length:
                    break
            for indices in generate_indices(frame_number, temp_size, temp_stride):
                self.indices.append([int(video), indices])
            if video_number >= max_video_number:
                break
        print('> Done. Loaded %d video sequences.' % len(self.data.keys()))

    def __getitem__(self, index):
        name, indices = self.indices[index]
        frame_names = self.data[name]

        path_compressed_frames, path_gt_frames = [], []
        for index in indices:
            path_compressed_frames.append(self.path_compressed_data + frame_names[index])
            path_gt_frames.append(self.path_gt_data + frame_names[index])

        compressed_frames = []
        for frame_path in path_compressed_frames:
            compressed = torch.from_numpy(np.float32(cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB))
                                          / 255).permute(2, 0, 1)
            compressed_frames.append(compressed)
        compressed_frames = torch.cat([frame for frame in compressed_frames], dim=0)

        path_gt_frame = path_gt_frames[self.temp_size // 2]
        gt_frame = cv2.cvtColor(cv2.imread(path_gt_frame), cv2.COLOR_BGR2RGB)
        if not self.sr:
            gt_frame = cv2.resize(gt_frame, (compressed_frames.shape[2], compressed_frames.shape[1]))
        gt_frame = torch.from_numpy(np.float32(gt_frame) / 255).permute(2, 0, 1)

        if not self.is_test:  # augment sequences if not test
            compressed_frames, gt_frame = self.augment_seq(compressed_frames, gt_frame)

        return compressed_frames, gt_frame

    def mirror_image(self, im):
        return tv.transforms.functional.hflip(im)

    def crop_image(self, im, x, y, is_gt=False):
        if not is_gt:
            im = im[:, x:x + self.input_patch_size, y:y + self.input_patch_size]
        else:
            im = im[:, x * 4:(x + self.input_patch_size) * 4, y * 4:(y + self.input_patch_size) * 4]
        return im

    def flip_image(self, im):
        return tv.transforms.functional.vflip(im)

    def augment_seq(self, compressed, gt):
        # data augmentation
        flip = False
        mirror = False
        if random.random() > 0.5:
            flip = True
        if random.random() > 0.5:
            mirror = True
        # crop param
        h = random.randint(0, compressed.shape[1] - self.input_patch_size)
        w = random.randint(0, compressed.shape[2] - self.input_patch_size)

        # apply augmentation to both compressed and gt frames
        compressed = self.crop_image(compressed, h, w)
        if self.sr:
            gt = self.crop_image(gt, h, w, is_gt=True)
        else:
            gt = self.crop_image(gt, h, w, is_gt=False)
        if flip:
            compressed = self.flip_image(compressed)
            gt = self.flip_image(gt)
        if mirror:
            compressed = self.mirror_image(compressed)
            gt = self.mirror_image(gt)
        return compressed, gt

    def __len__(self):
        return len(self.indices)
