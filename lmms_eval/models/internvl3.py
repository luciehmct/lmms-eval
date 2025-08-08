import logging
import math
import json
import os
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("eval_logger")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    
    # Calculate video duration
    video_length = len(vr) / fps

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    # Calculate timestamps for each selected frame
    image_times = []
    for frame_index in frame_indices:
        timestamp = frame_index / fps
        image_times.append(timestamp)
    
    print(f"DEBUG: load_video - video_path: {video_path}, total_frames: {len(vr)}, requested_segments: {num_segments}, selected_indices: {len(frame_indices)}, video_length: {video_length:.2f}s")
    
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list, image_times, video_length


def split_model(model_name, num_layers=None):
    device_map = {}
    world_size = torch.cuda.device_count()
    if num_layers is None:
        # InternVL3-8B uses Qwen2.5-7B which has 28 layers
        num_layers = {
            "InternVL3-1B": 16,  # Based on Qwen2.5-0.5B
            "InternVL3-2B": 24,  # Based on Qwen2.5-1.5B
            "InternVL3-8B": 28,  # Based on Qwen2.5-7B
            "InternVL3-14B": 40, # Based on Qwen2.5-14B
            "InternVL3-38B": 64, # Based on Qwen2.5-32B
            "InternVL3-78B": 80, # Based on Qwen2.5-72B
        }.get(model_name.split("/")[-1], 28)  # Default to 28 for InternVL3-8B
    
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


@register_model("internvl3")
class InternVL3(lmms):
    """
    InternVL3-8B model implementation for lmms-eval
    
    InternVL3 is an advanced multimodal large language model with superior
    multimodal perception and reasoning capabilities, including video understanding.
    """
    
    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3-8B",
        modality: str = "image",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        num_frame: int = 32, 
        dynamic_image_size=False,
        use_temporal_context: bool = True,  # Enable enhanced temporal context by default
        num_layers=None,
        max_num: int = 6,  # Maximum number of image tiles
        **kwargs,
    ):
        super().__init__()

        self.path = pretrained
        self.num_frame = num_frame
        self.max_num = max_num
        self.use_temporal_context = use_temporal_context

        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size should be 1 for InternVL3, but got {batch_size}."
        self.batch_size_per_gpu = batch_size

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        self.dynamic_image_size = dynamic_image_size

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            device_map = split_model(pretrained, num_layers=num_layers)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # Load the model with trust_remote_code=True for InternVL3
        self._model = AutoModel.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=self.device_map
        ).eval()
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.path,
            trust_remote_code=True,
            use_fast=False  # InternVL3 recommendation
        )

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        self.modality = modality
        
        # Initialize debug log file
        self.debug_log_file = None
        if hasattr(self, 'use_temporal_context') and self.use_temporal_context:
            # Create debug log file in the current working directory
            self.debug_log_file = f"mammalps_debug_prompts_{os.getpid()}.jsonl"
            eval_logger.info(f"Debug prompts will be saved to: {self.debug_log_file}")

    @property
    def config(self):
        return self._model.config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="InternVL3 Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            pop_keys = []
            for k, v in gen_kwargs.items():
                if k not in DEFAULT_GEN_KWARGS:
                    pop_keys.append(k)

            for k in pop_keys:
                gen_kwargs.pop(k)

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            
            if self.modality == "image":
                if visuals:
                    visuals = [load_image(visual, max_num=self.max_num).to(torch.bfloat16).cuda() for visual in visuals]
                    pixel_values = torch.cat(visuals, dim=0)
                    num_patches_list = [visual.size(0) for visual in visuals]
                    image_tokens = ["<image>"] * len(visuals)
                    image_tokens = " ".join(image_tokens)
                    contexts = image_tokens + "\n" + contexts
                else:
                    pixel_values = None
                    num_patches_list = None
                    
                response, history = self.model.chat(
                    self.tokenizer, 
                    pixel_values, 
                    contexts, 
                    gen_kwargs, 
                    num_patches_list=num_patches_list, 
                    history=None, 
                    return_history=True
                )
                
            elif self.modality == "video":
                assert len(visuals) == 1, f"Only one video is supported, but got {len(visuals)} videos."
                video_path = visuals[0]
                pixel_values, num_patches_list, image_times, video_length = load_video(
                    video_path, 
                    num_segments=self.num_frame,
                    max_num=1  # Use 1 for video frames to avoid excessive memory usage
                )
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                
                # DEBUG: Print actual frame count and temporal info
                print(f"DEBUG: Video processing - requested frames: {self.num_frame}, actual frames: {len(num_patches_list)}, patches per frame: {num_patches_list}")
                print(f"DEBUG: Video temporal info - video_length: {video_length:.2f}s, frame_timestamps: {[f'{t:.2f}s' for t in image_times[:5]]}{'...' if len(image_times) > 5 else ''}")
                
                # Create enhanced video prefix with temporal information (animal_dataset.py style)
                if hasattr(self, 'use_temporal_context') and self.use_temporal_context:
                    # Enhanced version with timestamps and video duration
                    special_tokens = "\n".join([
                        "Frame-{} at second {:.2f}: <image>".format(i + 1, image_times[i])
                        for i in range(len(num_patches_list))
                    ])
                    video_prefix = (
                        "The video is {:.2f} second(s) long and you can see the frames below:\n".format(video_length)
                        + special_tokens + "\n"
                    )
                    # Replace <video> placeholder if present, otherwise prepend
                    if "<video>" in contexts:
                        question = contexts.replace("<video>\n", video_prefix)
                    else:
                        question = video_prefix + contexts
                else:
                    # Standard version (current implementation)
                    video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
                    question = video_prefix + contexts
                
                # DEBUG: Log the complete final prompt with timestamps that will be sent to the model
                print(f"DEBUG: ===== FINAL PROMPT TO MODEL =====")
                print(f"DEBUG: Video path: {video_path}")
                print(f"DEBUG: Video length: {video_length:.2f}s, Frames: {len(num_patches_list)}")
                print(f"DEBUG: Prompt length: {len(question)} chars")
                print(f"DEBUG: Full prompt:")
                print(f"DEBUG: ----START----")
                print(question)
                print(f"DEBUG: ----END----")
                print(f"DEBUG: ================================")
                
                response, history = self.model.chat(
                    self.tokenizer, 
                    pixel_values, 
                    question, 
                    gen_kwargs, 
                    num_patches_list=num_patches_list, 
                    history=None, 
                    return_history=True
                )
                
                # Save debug information to JSONL file
                if self.debug_log_file:
                    try:
                        # Get document information
                        doc = self.task_dict[task][split][doc_id]
                        
                        # Extract ground truth based on task
                        ground_truth = []
                        if 'action' in task:
                            ground_truth = doc.get('action', [])
                        elif 'activity' in task:
                            ground_truth = doc.get('activity', [])
                        elif 'animal' in task:
                            ground_truth = doc.get('animal', [])
                        
                        # Extract the filtered answer from the full response
                        import re
                        filtered_answer = []
                        
                        # Try to extract from "Final answer: ['item1', 'item2']" format
                        final_answer_match = re.search(r"Final answer:\s*(\[.*?\])", response, re.IGNORECASE | re.DOTALL)
                        if final_answer_match:
                            try:
                                filtered_answer = eval(final_answer_match.group(1))
                                if isinstance(filtered_answer, list):
                                    filtered_answer = [str(label).strip() for label in filtered_answer]
                            except Exception as e:
                                print(f"DEBUG: Failed to parse final answer: {e}")
                                filtered_answer = []
                        
                        # Fallback: try to extract any list-like structure
                        if not filtered_answer:
                            list_match = re.search(r"\[([^\]]+)\]", response)
                            if list_match:
                                content = list_match.group(1)
                                # Split by comma and clean up
                                filtered_answer = [item.strip().strip("'\"") for item in content.split(",")]
                        
                        # Create debug entry
                        debug_entry = {
                            "dataset": "mammalps",
                            "video_id": doc.get('clip', 'unknown'),
                            "question_id": doc.get('id', doc_id),
                            "task": task,
                            "video_length_seconds": video_length,
                            "num_frames": len(num_patches_list),
                            "frame_timestamps": image_times,
                            "question": question,
                            "answer": filtered_answer,  # Extracted final answer list
                            "ground_truth": ground_truth,
                            "full_answer": response  # Complete model response with reasoning
                        }
                        
                        # Append to JSONL file
                        with open(self.debug_log_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(debug_entry, ensure_ascii=False) + '\n')
                        
                        print(f"DEBUG: Saved prompt info to {self.debug_log_file}")
                        
                    except Exception as e:
                        print(f"DEBUG: Failed to save debug info: {e}")
                
            else:
                raise ValueError(f"Unsupported modality: {self.modality}")
                
            res.append(response)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not implemented yet."

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for InternVL3")
