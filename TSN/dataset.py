import os
import pandas as pd
import numpy as np
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import torch
from torchvision.io import read_image

class UCF101(Dataset):
    def __init__(self, annotations_file, video_dir, flow_dir, validation=False, transform=None, target_transform=None):
        self.video_labels = pd.read_csv(annotations_file)
        self.video_dir = video_dir
        self.flow_dir = flow_dir
        self.validation = validation
        self.transform = transform
        self.target_transform = target_transform

        # get num of classes
        self.classnames = sorted(self.video_labels.iloc[:,0].str.split('/').str[0].unique())
        self.class_to_idx = {name: i for i, name in enumerate(self.classnames)}


    def __len__(self):
        return len(self.video_labels)

    def __getitem__(self, idx):
        # Load rgb video frames
        video_path = os.path.join(self.video_dir, self.video_labels.iloc[idx, 0])
        vr = VideoReader(video_path + ".avi", ctx=cpu(0))

        # getting frames for validation and for training
        num_frames = len(vr)
        num_frames_chunk = num_frames / 4
        frames_start_idx = np.array([0, num_frames_chunk, num_frames_chunk*2, num_frames_chunk*3])
        if self.validation:
            frames_start_idx += num_frames_chunk / 2
        else:
            idx_offset = np.random.randint(low=0, high=num_frames/4, size=4)
            frames_start_idx += idx_offset
            
        frames = vr.get_batch(frames_start_idx).asnumpy()
        frames = torch.from_numpy(frames).float() / 255.0    
        frames = frames.permute(0, 3, 1, 2) 

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames = (frames - mean) / std 

        # load flow frames
        video_flow_dir = self.flow_dir + "/" + self.video_labels.iloc[idx, 0]
        num_flows = len(os.listdir(video_flow_dir)) / 2
        num_flow_chunks = num_flows / 4
        flow_start_idx = np.array([0, num_flow_chunks, num_flow_chunks*2, num_flow_chunks*3])
        if self.validation:
            flow_start_idx += num_flow_chunks / 2
        else:
            idx_offset = np.random.randint(low=0, high=(num_flows/4) - 7, size=4)
            flow_start_idx += idx_offset

        flows_snipets = []
        for start in flow_start_idx:
            s = int(start) + 1
            paths_x = [video_flow_dir + "/flow_x_" + f"{i:04d}.jpg" for i in range(int(s), int(s+7))]
            paths_y = [video_flow_dir + "/flow_y_" + f"{i:04d}.jpg" for i in range(int(s), int(s+7))]
            fx = torch.stack([read_image(p) for p in paths_x]).squeeze(1)
            fy = torch.stack([read_image(p) for p in paths_y]).squeeze(1)
            snippet = torch.stack([fx, fy], dim=1)
            snippet = snippet.reshape(2 * 7, *fx.shape[1:])  
            flows_snipets.append(snippet)

        flows = torch.stack(flows_snipets).float()
        flows = (flows - 128.0) / 128.0

        label = self.class_to_idx[self.video_labels.iloc[idx, 0].split("/")[0]]
        if self.transform:
            frames = self.transform(frames)
        if self.target_transform:
            label = self.target_transform(label)
        return (frames, flows), label