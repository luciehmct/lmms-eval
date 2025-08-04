import json
import os
import torch
import logging
import base64
import cv2
import numpy as np
import torch.distributed as dist

from utils.internvl_utils import load_video as internvl_load_video
from internvl.train.dataset import build_transform


class VQADataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dir,
        test,
        prompt,
        cfg,
        input_size=448,
        dynamic_image_size=False,
        use_thumbnail=False,
        max_num=6,
    ):
        try:
            self.test = open(os.path.join(cfg.data_path, data_dir, test)).readlines()
        except UnicodeDecodeError as e:
            content = open(os.path.join(cfg.data_path, data_dir, test)).read()
            print(content[180:190])
            raise UnicodeDecodeError(e)
        logging.info(f"Loaded {len(self.test)} samples from {test}")
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num

        self.transform = build_transform(is_train=False, input_size=input_size)
        self.cfg = cfg
        self.data_dir = data_dir

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        video_id = data["video"]
        question = data["conversations"][0]["value"]
        question_id = data["id"]
        annotation = data["conversations"][-1]["value"]

        video_path = os.path.join(self.cfg.data_path, self.data_dir, video_id)

        image_list, num_patches_list, image_times, video_length = internvl_load_video(
            video_path, question, self.cfg
        )

        if "gpt-" in self.cfg.videollm.checkpoint:
            question = question.replace("<video>", "")

            special_tokens = [
                "Frame-{} at second {:.2f}: <image>".format(i + 1, image_times[i])
                for i in range(len(image_list))
            ]

            processed_image_list = []
            for frame_index, frame in enumerate(image_list):
                frame = frame.permute(1, 2, 0).detach().cpu().numpy()
                if frame.dtype != np.uint8:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode(".jpg", frame_bgr)
                encoded_frame = base64.b64encode(buffer).decode("utf-8")
                processed_image_list.append(encoded_frame)
            image_list = processed_image_list

        else:
            special_tokens = "\n".join(
                [
                    "Frame-{} at second {:.2f}: <image>".format(i + 1, image_times[i])
                    for i in range(len(image_list))
                ]
            )
            special_tokens = (
                "The video is {:.2f} second(s) long and you can see the frames below:\n".format(
                    video_length
                )
                + special_tokens
            )

            question = question.replace("<video>\n", special_tokens + "\n")

        if len(self.prompt) != 0:
            question = question + "\n\n" + self.prompt
        return {
            "video_id": video_id,
            "video_length": video_length,
            "question_id": question_id,
            "question": question,
            "pixel_values": image_list,
            "frame_info": special_tokens,
            "annotation": annotation,
        }


def collate_fn(batches):
    pixel_values = (
        torch.cat([_["pixel_values"] for _ in batches], dim=0)
        if type(batches[0]["pixel_values"]) is torch.Tensor
        else [_["pixel_values"] for _ in batches]
    )
    frame_infos = [_["frame_info"] for _ in batches]
    questions = [_["question"] for _ in batches]
    question_ids = [_["question_id"] for _ in batches]
    annotations = [_["annotation"] for _ in batches]
    video_ids = [_["video_id"] for _ in batches]
    video_lengths = [_["video_length"] for _ in batches]

    return (
        video_ids,
        video_lengths,
        pixel_values,
        frame_infos,
        questions,
        question_ids,
        annotations,
    )


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(
            size, self._world_size, self._rank
        )

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
