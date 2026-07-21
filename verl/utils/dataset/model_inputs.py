# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared tokenizer/processor input construction for RL and agent rollouts."""

from __future__ import annotations

import copy
import math
import numbers
import re

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


def as_media_list(value):
    """Normalize parquet media cells without treating an image dict as an iterable."""
    if value is None:
        return []
    if isinstance(value, numbers.Real) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if value.__class__.__module__.startswith("numpy") and hasattr(value, "tolist"):
        return as_media_list(value.tolist())
    return [value]


def build_multimodal_messages(messages, image_count: int = 0, video_count: int = 0):
    """Convert textual media placeholders into processor-native chat segments."""
    messages = copy.deepcopy(messages)
    found_images = 0
    found_videos = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            found_images += sum(segment.get("type") == "image" for segment in content if isinstance(segment, dict))
            found_videos += sum(segment.get("type") == "video" for segment in content if isinstance(segment, dict))
            continue
        if not isinstance(content, str):
            continue
        segments = []
        for segment in re.split("(<image>|<video>)", content):
            if segment == "<image>":
                segments.append({"type": "image"})
                found_images += 1
            elif segment == "<video>":
                segments.append({"type": "video"})
                found_videos += 1
            elif segment:
                segments.append({"type": "text", "text": segment})
        message["content"] = segments

    if image_count and found_images == 0:
        user_message = next((message for message in reversed(messages) if message.get("role") == "user"), None)
        if user_message is None:
            raise ValueError("Cannot attach an image to a chat without a user message.")
        user_message["content"] = [{"type": "image"} for _ in range(image_count)] + user_message["content"]
        found_images = image_count
    if video_count and found_videos == 0:
        user_message = next((message for message in reversed(messages) if message.get("role") == "user"), None)
        if user_message is None:
            raise ValueError("Cannot attach a video to a chat without a user message.")
        user_message["content"] = [{"type": "video"} for _ in range(video_count)] + user_message["content"]
        found_videos = video_count

    if found_images != image_count or found_videos != video_count:
        raise ValueError(
            "Media placeholder count does not match supplied data: "
            f"images={found_images}/{image_count}, videos={found_videos}/{video_count}."
        )
    return messages


def process_multimodal_chat(
    *,
    messages,
    tokenizer,
    processor,
    images=None,
    videos=None,
    max_length: int,
    truncation: str,
):
    """Build padded HF training inputs and unexpanded vLLM prompt IDs."""
    images = list(images or [])
    videos = list(videos or [])
    if processor is None:
        raise ValueError("A Hugging Face processor is required when images or videos are present.")

    processor_messages = build_multimodal_messages(messages, len(images), len(videos))
    raw_prompt = processor.apply_chat_template(processor_messages, add_generation_prompt=True, tokenize=False)
    processor_kwargs = {"text": [raw_prompt], "return_tensors": "pt"}
    if images:
        processor_kwargs["images"] = images
    if videos:
        processor_kwargs["videos"] = videos
    model_inputs = processor(**processor_kwargs)

    input_ids = model_inputs.pop("input_ids")
    attention_mask = model_inputs.pop("attention_mask")
    if input_ids.shape[-1] > max_length:
        raise RuntimeError(
            f"Expanded multimodal prompt length {input_ids.shape[-1]} exceeds max_length={max_length}; "
            "increase max_length because truncating image/video tokens breaks feature alignment."
        )
    raw_attention_mask = attention_mask.clone()
    rope_inputs = dict(model_inputs)
    model_inputs.pop("second_per_grid_ts", None)

    input_ids, attention_mask = verl_F.postprocess_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation=truncation,
    )

    for key in ("token_type_ids", "mm_token_type_ids"):
        if key not in model_inputs:
            continue
        model_inputs[key], _ = verl_F.postprocess_data(
            input_ids=model_inputs[key],
            attention_mask=raw_attention_mask,
            max_length=max_length,
            pad_token_id=0,
            left_pad=True,
            truncation=truncation,
        )

    image_processor_name = getattr(getattr(processor, "image_processor", None), "__class__", type(None)).__name__
    if image_processor_name.startswith("Qwen2"):
        from verl.models.transformers.qwen2_vl import get_rope_index

        position_ids = [
            get_rope_index(
                processor,
                input_ids=input_ids[0],
                image_grid_thw=rope_inputs.get("image_grid_thw"),
                video_grid_thw=rope_inputs.get("video_grid_thw"),
                second_per_grid_ts=rope_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )
        ]
    else:
        position_ids = compute_position_id_with_mask(attention_mask)

    raw_prompt_ids = tokenizer.encode(raw_prompt, add_special_tokens=False)
    if len(raw_prompt_ids) > max_length:
        if truncation == "left":
            raw_prompt_ids = raw_prompt_ids[-max_length:]
        elif truncation == "right":
            raw_prompt_ids = raw_prompt_ids[:max_length]
        elif truncation == "middle":
            left_half = max_length // 2
            raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-(max_length - left_half) :]
        elif truncation == "error":
            raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {max_length}.")

    multi_modal_data = {}
    if images:
        multi_modal_data["image"] = images
    if videos:
        multi_modal_data["video"] = [video.numpy() if hasattr(video, "numpy") else video for video in videos]

    return {
        "input_ids": input_ids[0],
        "attention_mask": attention_mask[0],
        "position_ids": position_ids[0],
        "raw_prompt_ids": raw_prompt_ids,
        "raw_prompt": raw_prompt,
        "multi_modal_data": multi_modal_data,
        "multi_modal_inputs": dict(model_inputs),
    }
