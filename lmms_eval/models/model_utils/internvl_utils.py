import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu


def adaptive_keyframe_sampling(video_path: str, num_segments: int, query: str) -> np.ndarray:
    """
    Select frame indices relevant to the textual query using CLIP similarity.
    Falls back to uniform sampling if CLIP loading fails due to PyTorch version issues.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    num_segments : int
        Number of key frames to sample.
    query : str
        Text query describing the desired content.

    Returns
    -------
    np.ndarray
        Sorted array of selected frame indices.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    # Fallback to uniform indices when query is empty
    if not query or not query.strip():
        return np.linspace(0, total_frames - 1, num_segments, dtype=int)

    try:
        # Try to load CLIP with safetensors support
        from transformers import CLIPModel, CLIPProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Try multiple approaches to load CLIP safely
        model = None
        processor = None

        try:
            model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                use_safetensors=True,
            ).to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
            load_msg = "✓ CLIP loaded with safetensors"
        except Exception as e1:
            try:
                model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch16",
                    use_safetensors=True,
                ).to(device)
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=True)
                load_msg = "✓ CLIP loaded with clip-vit-base-patch16 and safetensors"
            except Exception as e2:
                try:
                    model = CLIPModel.from_pretrained(
                        "openai/clip-vit-base-patch32",
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        trust_remote_code=False,
                    ).to(device)
                    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
                    load_msg = "✓ CLIP loaded with float16 and safetensors"
                except Exception as e3:
                    print("All CLIP loading attempts failed:")
                    print(f"  Attempt 1 (safetensors): {e1}")
                    print(f"  Attempt 2 (patch16): {e2}")
                    print(f"  Attempt 3 (float16): {e3}")
                    print("Falling back to uniform sampling")
                    return np.linspace(0, total_frames - 1, num_segments, dtype=int)

        raw_tokens = processor.tokenizer(query)["input_ids"]
        text_inputs = processor(text=query, return_tensors="pt", truncation=True, max_length=77).to(device)
        if len(raw_tokens) > text_inputs["input_ids"].shape[-1]:
            print("!! query truncated to 77 tokens for CLIP")
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        candidate_indices = np.linspace(0, total_frames - 1, num_segments * 4, dtype=int)
        frames = [Image.fromarray(vr[idx].asnumpy()) for idx in candidate_indices]
        img_inputs = processor(images=frames, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**img_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        print(load_msg)
        scores = (image_features @ text_features.T).squeeze()
        topk = scores.topk(num_segments).indices.cpu().numpy()
        selected = np.sort(candidate_indices[topk])

        print(f"✓ Adaptive sampling successful: selected {len(selected)} frames")
        return selected

    except ImportError as e:
        print(f"CLIP dependencies not available: {e}")
        print("Falling back to uniform sampling")
        return np.linspace(0, total_frames - 1, num_segments, dtype=int)

    except Exception as e:
        print(f"Adaptive sampling failed with error: {e}")
        print("Falling back to uniform sampling")
        return np.linspace(0, total_frames - 1, num_segments, dtype=int)
