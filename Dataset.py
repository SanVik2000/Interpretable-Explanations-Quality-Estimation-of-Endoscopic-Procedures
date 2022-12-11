import os
import time
import torch
import os.path
import fnmatch
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import List, Union
from torchvision import transforms
from torch.utils.data import DataLoader

def get_frame_count(dir_path):
    return len(fnmatch.filter(os.listdir(dir_path), '*.png'))

def get_label_from_file(file_path):
    f = open(file_path, "r")
    file_contents = f.read()
    if 'correct' in file_contents:
        return 1
    else:
        return 0 

class VideoRecord(object):
    def __init__(self, item, root_path, label, num_frames):
        self._num_frames = num_frames
        self._label = label
        self._path = os.path.join(item)

    @property
    def path(self) -> str:
        return self._path

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def start_frame(self) -> int:
        return int(0)

    @property
    def end_frame(self) -> int:
        return int(self._num_frames)

    @property
    def label(self) -> Union[int, List[int]]:
        return self._label

class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_path: str,
                 num_segments: int = 1,
                 frames_per_segment: int = 100,
                 transform = None,
                 phase = 'train',
                 vis: bool = False):
        super(VideoFrameDataset, self).__init__()

        self.root_path = root_path
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.transform = transform
        self.phase = phase
        self.vis = vis

        self.imagefile_template = 'Image{:0d}.png'

        self._parse_videos()
        self._sanity_check_samples()

    def get_labels(self):
        return self.label_list
        
    def _load_image(self, directory, idx):
        return Image.open(os.path.join(directory, 'Image'+str(idx)+'.png')).convert('RGB')

    def _parse_videos(self):
        subfolders = [ f.path for f in os.scandir(self.root_path) if f.is_dir() ]
        self.video_list = []
        self.label_list = []
        for item in tqdm(subfolders):
            label = get_label_from_file(os.path.join(item, 'label.txt'))
            num_frames = get_frame_count(os.path.join(item))
            #print(item , num_frames, label)
            self.video_list.append(VideoRecord(item, self.root_path, label, num_frames))
            self.label_list.append(label)

    def _sanity_check_samples(self):
        for record in self.video_list:
            if record.num_frames <= 0 or record.start_frame == record.end_frame:
                print(f"\nDataset Warning: video {record.path} seems to have zero RGB frames on disk!\n")

            elif record.num_frames < (self.num_segments * self.frames_per_segment):
                print(f"\nDataset Warning: video {record.path} has {record.num_frames} frames "
                      f"but the dataloader is set up to load "
                      f"(num_segments={self.num_segments})*(frames_per_segment={self.frames_per_segment})"
                      f"={self.num_segments * self.frames_per_segment} frames. Dataloader will throw an "
                      f"error when trying to load this video.\n")

    def _get_start_indices(self, record: VideoRecord) -> 'np.ndarray[int]':
        # choose start indices that are perfectly evenly spread across the video frames if validation.
        if self.phase == 'validation':
            distance_between_indices = (record.num_frames - self.frames_per_segment) / float(self.num_segments)
            start_indices = np.array([int(distance_between_indices / 2.0 + distance_between_indices * x) for x in range(self.num_segments)])
        
        # choose 0 as start_index if test.
        if self.phase == 'test':
            distance_between_indices = (record.num_frames - self.frames_per_segment)
            start_indices = np.array([0])

        # randomly sample start indices that are approximately evenly spread across the video frames if train.
        else:
            max_valid_start_index = (record.num_frames - self.frames_per_segment)
            start_indices = np.random.randint(max_valid_start_index, size=self.num_segments)

        return start_indices

    def __getitem__(self, idx):
        record = self.video_list[idx]

        frame_start_indices = self._get_start_indices(record)

        return self._get(record, frame_start_indices)

    def _get(self, record, frame_start_indices):

        frame_start_indices = frame_start_indices + record.start_frame
        images = list()

        # from each start_index, load self.frames_per_segment
        # consecutive frames
        for start_index in frame_start_indices:
            frame_index = int(start_index)

            # load self.frames_per_segment consecutive frames
            for _ in range(self.frames_per_segment):
                image = self._load_image(record.path, frame_index)
                images.append(image)

                if frame_index < record.end_frame:
                    frame_index += 1

        if self.transform is not None:
            transformed_images = self.transform(images)
            return transformed_images, record.label

        else:
            return images, record.label

    def __len__(self):
        return len(self.video_list)

class ImglistToTensor(torch.nn.Module):
    @staticmethod
    def forward(img_list):
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])

#----------------Transform----------------
def get_transform(mode='train'):
    if mode == 'train':
        return transforms.Compose([ImglistToTensor(), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)])
    elif mode =='val' or mode == 'test':
        return transforms.Compose([ImglistToTensor(), transforms.Resize(256), transforms.CenterCrop(224)])

#------------------Data_Module------------------
class DataModule:
    def __init__(self, mode = 'train', args=None):
        super().__init__()
        self.mode = mode
        self.batch_size = args.batch_size
        self.args = args
        
        self.train_transform = get_transform(mode='train')
        self.val_transform = get_transform(mode='val')
        self.test_transform = get_transform(mode='test')
        
    def prepare_data(self):
        
        if self.mode == 'train':
            start = time.time()

            self.train_dataset = VideoFrameDataset(root_path='../Dataset/' + 'train', num_segments=1, frames_per_segment=self.args.frame_count, transform=self.train_transform, phase='train')
            self.val_dataset = VideoFrameDataset(root_path='../Dataset/' + 'validation', num_segments=1, frames_per_segment=self.args.frame_count, transform=self.val_transform, phase='test')
            
            print("Number of train samples : " , len(self.train_dataset))
            print("Number of Val Samples : " , len(self.val_dataset))
            print("Dataset Loaded in Time : " , time.time() - start)
        
        elif self.mode == 'test':
            if self.args.vis:
                self.test_dataset = VideoFrameDataset(root_path='../Dataset/' + 'validation', num_segments=1, frames_per_segment=self.args.frame_count, transform=None, phase='test')
            else:
                self.test_dataset = VideoFrameDataset(root_path='../Dataset/' + 'validation', num_segments=1, frames_per_segment=self.args.frame_count, transform=self.test_transform, phase='test')
            
            print("Number of test samples : " , len(self.test_dataset))


    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False, num_workers=self.args.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=self.args.num_workers)

if __name__ == "__main__":
    train_dataset = VideoFrameDataset(root_path='/media/sanvik/Data/Dual_Degree_Project/' + 'train', num_segments=1, frames_per_segment=100, transform=None, test_mode=False)
