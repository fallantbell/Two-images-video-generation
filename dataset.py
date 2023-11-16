# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional
from zipfile import ZipFile

import einops
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from color import *

@dataclass
class VideoDataset_ACID(Dataset):
    dataset_dir: str
    seq_length: int
    height: int
    width: int
    min_spacing: int = 1
    max_spacing: int = 1
    min_video_length: Optional[int] = None
    x_flip: bool = False

    def __post_init__(self):
        assert self.seq_length >= 1

        self.dataset_path = Path(self.dataset_dir)
        assert self.dataset_path.is_dir(), self.dataset_path

        self.min_video_length = max(self.min_video_length or 1, (self.seq_length - 1) * self.min_spacing + 1)

        self.video_paths=[]

        for dir in os.listdir(self.dataset_dir):
            dir_name = self.dataset_dir+"/"+dir
            frame_names = [] 
            for img in os.listdir(dir_name):
                frame_names.append(img)
            
            if len(frame_names) >= self.min_video_length:
                video_tuple = (dir_name,frame_names)
                self.video_paths.append(video_tuple)
        
        print_g(f"availabe training data:{len(self.video_paths)}") 

        # self.video_paths = [
        #     (partition_name, clip_path, frame_names)
        #     for partition_name, partition_frame_paths in sorted(self.frame_paths.items())
        #     for clip_path, frame_names in sorted(partition_frame_paths.items())
        #     if len(frame_names) >= self.min_video_length
        # ]

        self._zipfiles = {}

    def sample_frame_names(self, frame_names: list[str]) -> tuple[list[str], int]:
        max_spacing = (
            1 if self.seq_length == 1 else min(self.max_spacing, (len(frame_names) - 1) // (self.seq_length - 1))
        )
        spacing = torch.randint(self.min_spacing, max_spacing + 1, size=()).item()

        frame_span = (self.seq_length - 1) * spacing + 1
        max_start_index = len(frame_names) - frame_span
        start_index = torch.randint(max_start_index + 1, size=()).item()

        frame_names = frame_names[start_index : start_index + frame_span : spacing]
        return frame_names, spacing
    
    def center_crop_and_resize(self,frame: Image, height: int, width: int) -> Image:
        #* 去除inf nature 黑邊
        frame = frame.crop((0,(1/5)*frame.height,frame.width,(4/5)*frame.height))

        # Measures by what factor height and width are larger/smaller than desired.
        height_scale = frame.height / height
        width_scale = frame.width / width

        # Center crops whichever dimension has a greater scale factor.
        if height_scale > width_scale:
            crop_height = height * width_scale
            y0 = (frame.height - crop_height) // 2
            y1 = y0 + crop_height
            frame = frame.crop((0, y0, frame.width, y1))

        elif width_scale > height_scale:
            crop_width = width * height_scale
            x0 = (frame.width - crop_width) // 2
            x1 = x0 + crop_width
            frame = frame.crop((x0, 0, x1, frame.height))

        # Resizes to desired height and width.
        frame = frame.resize((width, height), Image.LANCZOS)
        return frame

    def read_frame(self, frame_path: str) -> torch.Tensor:
        # if partition_name not in self._zipfiles:
        #     partition_path = self.dataset_path.joinpath(f"{partition_name}.zip")
        #     self._zipfiles[partition_name] = ZipFile(partition_path)

        # with self._zipfiles[partition_name].open(frame_path, "r") as fp:
        #     frame = np.array(Image.open(fp))

        frame = Image.open(frame_path)
        frame = self.center_crop_and_resize(frame,self.height,self.width)
        frame.save(f"test_folder/test.png")
        frame = np.array(frame)
        frame = torch.from_numpy(frame)
        frame = einops.rearrange(frame, "h w c -> c h w")
        frame = 2 * frame.to(torch.float32) / 255 - 1
        return frame

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip_path, frame_names = self.video_paths[index]
        frame_names, spacing = self.sample_frame_names(frame_names)
        frame_paths = [str(PurePosixPath(clip_path).joinpath(frame_name)) for frame_name in frame_names]
        frames = [self.read_frame(frame_path) for frame_path in frame_paths]
        video = torch.stack(frames, dim=1)

        if self.x_flip and torch.rand(()).item() < 0.5:
            video = video.flip(dims=(-1,))

        return dict(video=video, spacing=spacing)

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getstate__(self):
        return dict(self.__dict__, _zipfiles={})


# =====================================================================================================================


@dataclass
class VideoDataset(Dataset):
    dataset_dir: str
    seq_length: int
    height: int
    width: int
    min_spacing: int = 1
    max_spacing: int = 1
    min_video_length: Optional[int] = None
    x_flip: bool = False

    def __post_init__(self):
        assert self.seq_length >= 1

        self.dataset_path = Path(self.dataset_dir).joinpath(f"{self.height:04d}x{self.width:04d}")
        assert self.dataset_path.is_dir(), self.dataset_path

        self.frame_paths = {}
        for partition in self.dataset_path.glob("*.zip"):
            with ZipFile(partition) as zf:
                with zf.open("frame_paths.json", "r") as fp:
                    self.frame_paths[partition.stem] = json.load(fp)

        self.min_video_length = max(self.min_video_length or 1, (self.seq_length - 1) * self.min_spacing + 1)

        self.video_paths = [
            (partition_name, clip_path, frame_names)
            for partition_name, partition_frame_paths in sorted(self.frame_paths.items())
            for clip_path, frame_names in sorted(partition_frame_paths.items())
            if len(frame_names) >= self.min_video_length
        ]

        self._zipfiles = {}

    def sample_frame_names(self, frame_names: list[str]) -> tuple[list[str], int]:
        max_spacing = (
            1 if self.seq_length == 1 else min(self.max_spacing, (len(frame_names) - 1) // (self.seq_length - 1))
        )
        spacing = torch.randint(self.min_spacing, max_spacing + 1, size=()).item()

        frame_span = (self.seq_length - 1) * spacing + 1
        max_start_index = len(frame_names) - frame_span
        start_index = torch.randint(max_start_index + 1, size=()).item()

        frame_names = frame_names[start_index : start_index + frame_span : spacing]
        return frame_names, spacing

    def read_frame(self, partition_name: str, frame_path: str) -> torch.Tensor:
        if partition_name not in self._zipfiles:
            partition_path = self.dataset_path.joinpath(f"{partition_name}.zip")
            self._zipfiles[partition_name] = ZipFile(partition_path)

        with self._zipfiles[partition_name].open(frame_path, "r") as fp:
            frame = np.array(Image.open(fp))

        frame = torch.from_numpy(frame)
        frame = einops.rearrange(frame, "h w c -> c h w")
        frame = 2 * frame.to(torch.float32) / 255 - 1
        return frame

    def __getitem__(self, index: int) -> dict[str, Any]:
        partition_name, clip_path, frame_names = self.video_paths[index]
        frame_names, spacing = self.sample_frame_names(frame_names)
        frame_paths = [str(PurePosixPath(clip_path).joinpath(frame_name)) for frame_name in frame_names]
        frames = [self.read_frame(partition_name, frame_path) for frame_path in frame_paths]
        video = torch.stack(frames, dim=1)

        if self.x_flip and torch.rand(()).item() < 0.5:
            video = video.flip(dims=(-1,))

        return dict(video=video, spacing=spacing)

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getstate__(self):
        return dict(self.__dict__, _zipfiles={})


# =====================================================================================================================


@dataclass
class VideoDatasetTwoRes(Dataset):
    dataset_dir: str
    seq_length: int
    lr_height: int
    lr_width: int
    hr_height: int
    hr_width: int
    min_spacing: int = 1
    max_spacing: int = 1
    min_video_length: Optional[int] = None
    x_flip: bool = False

    def __post_init__(self):
        self.lr_dataset = VideoDataset(
            self.dataset_dir,
            self.seq_length,
            self.lr_height,
            self.lr_width,
            self.min_spacing,
            self.max_spacing,
            self.min_video_length,
            x_flip=self.x_flip,
        )
        self.hr_dataset = VideoDataset(
            self.dataset_dir,
            self.seq_length,
            self.hr_height,
            self.hr_width,
            self.min_spacing,
            self.max_spacing,
            self.min_video_length,
            x_flip=self.x_flip,
        )
        assert self.lr_dataset.video_paths == self.hr_dataset.video_paths

    def __getitem__(self, index: int) -> dict[str, Any]:
        partition_name, clip_path, frame_names = self.lr_dataset.video_paths[index]
        frame_names, spacing = self.lr_dataset.sample_frame_names(frame_names)
        frame_paths = [str(PurePosixPath(clip_path).joinpath(frame_name)) for frame_name in frame_names]
        lr_frames = [self.lr_dataset.read_frame(partition_name, frame_path) for frame_path in frame_paths]
        hr_frames = [self.hr_dataset.read_frame(partition_name, frame_path) for frame_path in frame_paths]
        lr_video = torch.stack(lr_frames, dim=1)
        hr_video = torch.stack(hr_frames, dim=1)

        if self.x_flip and torch.rand(()).item() < 0.5:
            lr_video = lr_video.flip(dims=(-1,))
            hr_video = hr_video.flip(dims=(-1,))

        return dict(lr_video=lr_video, hr_video=hr_video, spacing=spacing)

    def __len__(self) -> int:
        return len(self.lr_dataset)


# =====================================================================================================================


@dataclass
class VideoDatasetPerImage(Dataset):
    dataset_dir: str
    height: int
    width: int
    seq_length: int = 1
    x_flip: bool = False

    def __post_init__(self):
        self.dataset = VideoDataset(self.dataset_dir, seq_length=1, height=self.height, width=self.width)

        self.video_paths = []
        for partition_name, partition_frame_paths in sorted(self.dataset.frame_paths.items()):
            for clip_path, frame_names in sorted(partition_frame_paths.items()):
                num_samples_from_source = len(frame_names) - self.seq_length + 1
                for start_index in range(0, num_samples_from_source):
                    sample_frame_names = frame_names[start_index : start_index + self.seq_length]
                    self.video_paths.append((partition_name, clip_path, sample_frame_names, num_samples_from_source))

    def __getitem__(self, index: int) -> dict[str, Any]:
        partition_name, clip_path, sample_frame_names, num_samples_from_source = self.video_paths[index]
        frame_paths = [str(PurePosixPath(clip_path).joinpath(frame_name)) for frame_name in sample_frame_names]
        frames = [self.dataset.read_frame(partition_name, frame_path) for frame_path in frame_paths]
        video = torch.stack(frames, dim=1)

        if self.x_flip and torch.rand(()).item() < 0.5:
            video = video.flip(dims=(-1,))

        return dict(video=video, num_samples_from_source=num_samples_from_source)

    def __len__(self) -> int:
        return len(self.video_paths)


# =====================================================================================================================
