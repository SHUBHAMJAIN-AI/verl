# # Copyright 2024 Bytedance Ltd. and/or its affiliates
# # Copyright 2023-2024 SGLang Team
# # Copyright 2025 ModelBest Inc. and/or its affiliates
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import copy
# import logging
# import os
# import re
# import uuid
# from collections import defaultdict
# from typing import List, Optional, Union

# import datasets
# import numpy as np
# import torch
# from omegaconf import DictConfig, ListConfig
# from torch.utils.data import Dataset
# from transformers import PreTrainedTokenizer, ProcessorMixin

# import verl.utils.torch_functional as verl_F
# from verl.utils.model import compute_position_id_with_mask

# logger = logging.getLogger(__name__)


# def collate_fn(data_list: list[dict]) -> dict:
#     """
#     Collate a batch of sample dicts into batched tensors and arrays.

#     Args:
#         data_list: List of dicts mapping feature names to torch.Tensor or other values.

#     Returns:
#         Dict where tensor entries are stacked into a torch.Tensor of shape
#         (batch_size, *dims) and non-tensor entries are converted to
#         np.ndarray of dtype object with shape (batch_size,).
#     """
#     tensors = defaultdict(list)
#     non_tensors = defaultdict(list)

#     for data in data_list:
#         for key, val in data.items():
#             if isinstance(val, torch.Tensor):
#                 tensors[key].append(val)
#             else:
#                 non_tensors[key].append(val)

#     for key, val in tensors.items():
#         tensors[key] = torch.stack(val, dim=0)

#     for key, val in non_tensors.items():
#         non_tensors[key] = np.array(val, dtype=object)

#     return {**tensors, **non_tensors}


# class RLHFDataset(Dataset):
#     """
#     Load and preprocess RLHF data from Parquet files.

#     - Caches files locally.
#     - Reads into a HuggingFace Dataset and tokenizes prompts.
#     - Optionally handles images/videos via a ProcessorMixin.
#     - Filters prompts over a max length.
#     - Supports resuming from checkpoints.

#     Args:
#         data_files (str or list): Path(s) to Parquet file(s).
#         tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
#         config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
#         processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
#     """

#     def __init__(
#         self,
#         data_files: Union[str, List[str]],
#         tokenizer: PreTrainedTokenizer,
#         config: DictConfig,
#         processor: Optional[ProcessorMixin] = None,
#     ):
#         if not isinstance(data_files, (List, ListConfig)):
#             data_files = [data_files]

#         self.data_files = copy.deepcopy(data_files)
#         self.original_data_files = copy.deepcopy(data_files)  # use for resume
#         self.tokenizer = tokenizer
#         self.processor = processor
#         self.config = config

#         self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
#         self.prompt_key = config.get("prompt_key", "prompt")
#         self.image_key = config.get("image_key", "images")
#         self.video_key = config.get("video_key", "videos")
#         self.max_prompt_length = config.get("max_prompt_length", 1024)
#         self.return_raw_chat = config.get("return_raw_chat", False)
#         self.return_full_prompt = config.get("return_full_prompt", False)
#         self.truncation = config.get("truncation", "error")
#         self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

#         self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
#         self.num_workers = min(self.num_workers, os.cpu_count())
#         self.use_shm = config.get('use_shm', False)
#         self.chat_template_func = config.get("chat_template_func", None)
#         self.need_tools_kwargs = config.get("need_tools_kwargs", False)
#         self.filter_prompts = config.get("filter_prompts", True)
#         self.serialize_dataset = False
#         self.universal_id_key = config.get("universal_id_key", "universal_id")
#         # Dataset directory for permanent storage
#         self.dataset_dir = config.get("dataset_dir", "dataset")

#         self._download()
#         self._read_files_and_tokenize()

#     def _download(self, use_origin_parquet=False):
#         from verl.utils.fs import copy_to_local

#         data_files = self.data_files if not use_origin_parquet else self.original_data_files
#         for i, parquet_file in enumerate(data_files):
#             self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

#     def _read_files_and_tokenize(self):
#         dataframes = []
#         for parquet_file in self.data_files:
#             # read parquet files and cache
#             dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
#             dataframes.append(dataframe)
#         self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

#         print(f"dataset len: {len(self.dataframe)}")

#         # --- Add universal_id to each sample if not present ---
#         def add_universal_id(example):
#             uid_key = self.universal_id_key
#             if uid_key not in example or not example[uid_key]:
#                 example[uid_key] = str(uuid.uuid4())
#             # For backward compatibility, also set "universal_id" if different
#             if uid_key != "universal_id" and "universal_id" not in example:
#                 example["universal_id"] = example[uid_key]
#             return example

#         self.dataframe = self.dataframe.map(add_universal_id, desc="Adding universal_id to each sample")

#         print("Sample universal_ids after adding:")
#         for i in range(3):
#             print(self.dataframe[i].get("universal_id"), self.dataframe[i])

#         # Save to permanent dataset folder instead of cache
#         os.makedirs(self.dataset_dir, exist_ok=True)

#         # Save as both HuggingFace dataset and parquet
#         hf_save_dir = os.path.join(self.dataset_dir, "processed_with_uid_hf")
#         parquet_save_path = os.path.join(self.dataset_dir, "processed_with_uid.parquet")

#         print(f"Saving processed dataset with universal_id to {hf_save_dir}")
#         self.dataframe.save_to_disk(hf_save_dir)

#         print(f"Saving processed dataset as parquet to {parquet_save_path}")
#         self.dataframe.to_parquet(parquet_save_path)

#         # Also save to cache for compatibility
#         cache_save_dir = os.path.join(self.cache_dir, "processed_with_uid")
#         print(f"Also saving to cache: {cache_save_dir}")
#         self.dataframe.save_to_disk(cache_save_dir)

#         # filter out too long prompts
#         if self.filter_overlong_prompts:
#             tokenizer = self.tokenizer
#             prompt_key = self.prompt_key
#             self.dataframe = self.dataframe.filter(
#                 lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
#                 num_proc=self.num_workers,
#                 desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
#             )

#             print(f"filter dataset len: {len(self.dataframe)}")

#             # Also save filtered dataset
#             filtered_hf_dir = os.path.join(self.dataset_dir, "filtered_with_uid_hf")
#             filtered_parquet_path = os.path.join(self.dataset_dir, "filtered_with_uid.parquet")

#             print(f"Saving filtered dataset to {filtered_hf_dir}")
#             self.dataframe.save_to_disk(filtered_hf_dir)

#             print(f"Saving filtered dataset as parquet to {filtered_parquet_path}")
#             self.dataframe.to_parquet(filtered_parquet_path)

#     def resume_dataset_state(self):
#         self.serialize_dataset = not hasattr(self, "original_data_files")
#         # resume dataframe if not it's serialized in data.pt
#         if not self.serialize_dataset:
#             self._download(use_origin_parquet=True)  # download and resume from original parquet files
#             self._read_files_and_tokenize()
#         else:
#             print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

#     def __len__(self):
#         return len(self.dataframe)

#     def _build_messages(self, example: dict):
#         messages: list = example.pop(self.prompt_key)

#         if self.image_key in example or self.video_key in example:
#             for message in messages:
#                 content = message["content"]
#                 content_list = []
#                 for segment in re.split("(<image>|<video>)", content):
#                     if segment == "<image>":
#                         content_list.append({"type": "image"})
#                     elif segment == "<video>":
#                         content_list.append({"type": "video"})
#                     else:
#                         content_list.append({"type": "text", "text": segment})

#                 message["content"] = content_list

#         return messages

#     def __getitem__(self, item):
#         """
#         Note that we also return the raw_input_ids so that it can be combined with other chat template
#         """
#         row_dict: dict = self.dataframe[item]
#         messages = self._build_messages(row_dict)
#         model_inputs = {}

#         if self.processor is not None:
#             from verl.utils.dataset.vision_utils import process_image, process_video

#             raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
#             multi_modal_data = {}

#             images = None
#             if self.image_key in row_dict:
#                 images = [process_image(image) for image in row_dict.pop(self.image_key)]
#                 multi_modal_data["image"] = images

#             videos = None
#             if self.video_key in row_dict:
#                 videos = [process_video(video) for video in row_dict.pop(self.video_key)]
#                 multi_modal_data["video"] = [video.numpy() for video in videos]

#             model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

#             input_ids = model_inputs.pop("input_ids")
#             attention_mask = model_inputs.pop("attention_mask")

#             if "second_per_grid_ts" in model_inputs:
#                 model_inputs.pop("second_per_grid_ts")

#             # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
#             row_dict["multi_modal_data"] = multi_modal_data
#             row_dict["multi_modal_inputs"] = dict(model_inputs)

#             # second_per_grid_ts isn't used for training, just for mrope
#             row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

#         else:
#             raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
#             model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
#             input_ids = model_inputs.pop("input_ids")
#             attention_mask = model_inputs.pop("attention_mask")

#         input_ids, attention_mask = verl_F.postprocess_data(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             max_length=self.max_prompt_length,
#             pad_token_id=self.tokenizer.pad_token_id,
#             left_pad=True,
#             truncation=self.truncation,
#         )

#         if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
#             from verl.models.transformers.qwen2_vl import get_rope_index

#             position_ids = [
#                 get_rope_index(
#                     self.processor,
#                     input_ids=input_ids[0],
#                     image_grid_thw=model_inputs.get("image_grid_thw"),
#                     video_grid_thw=model_inputs.get("video_grid_thw"),
#                     second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
#                     attention_mask=attention_mask[0],
#                 )
#             ]  # (1, 3, seq_len)

#         else:
#             position_ids = compute_position_id_with_mask(attention_mask)

#         row_dict["input_ids"] = input_ids[0]
#         row_dict["attention_mask"] = attention_mask[0]
#         row_dict["position_ids"] = position_ids[0]

#         raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
#         if len(raw_prompt_ids) > self.max_prompt_length:
#             if self.truncation == "left":
#                 raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
#             elif self.truncation == "right":
#                 raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
#             elif self.truncation == "middle":
#                 left_half = self.max_prompt_length // 2
#                 right_half = self.max_prompt_length - left_half
#                 raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
#             elif self.truncation == "error":
#                 raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

#         row_dict["raw_prompt_ids"] = raw_prompt_ids
#         # encode prompts without chat template
#         if self.return_raw_chat:
#             row_dict["raw_prompt"] = messages

#         # get prompts with chat template
#         if self.return_full_prompt:
#             row_dict["full_prompts"] = raw_prompt  # array of strings

#         # add index for each prompt
#         index = row_dict.get("extra_info", {}).get("index", 0)
#         tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
#         need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
#         if need_tools_kwargs and not tools_kwargs:
#             logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
#         row_dict["index"] = index
#         row_dict["tools_kwargs"] = tools_kwargs

#         # --- Ensure the same universal_id is always used and never regenerated ---
#         uid_key = self.universal_id_key
#         # If neither uid_key nor "universal_id" is present, raise an error (should never happen if dataset is prepared correctly)
#         if uid_key not in row_dict and "universal_id" not in row_dict:
#             raise ValueError("universal_id missing from sample. This should not happen if dataset was prepared correctly.")
#         # If only "universal_id" is present, copy it to uid_key
#         if uid_key not in row_dict and "universal_id" in row_dict:
#             row_dict[uid_key] = row_dict["universal_id"]
#         # If both are present but different, always use the value from dataset (do not generate new)
#         # When logging or exporting, always use row_dict[uid_key] (never generate a new one)

#         return row_dict

#     def __getstate__(self):
#         if not self.serialize_dataset:
#             state = self.__dict__.copy()

#             if "dataframe" in state:
#                 del state["dataframe"]
#             return state

#         return self.__dict__.copy()

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.universal_id_key = config.get("universal_id_key", "universal_id")
        # Dataset directory for permanent storage
        self.dataset_dir = config.get("dataset_dir", "dataset")

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # --- Only check uid presence, do not generate ---
        uid_key = self.universal_id_key
        col_names = self.dataframe.column_names
        if uid_key not in col_names and "uid" not in col_names:
            raise ValueError("uid column is missing from the dataset! Please add it before loading.")

        print("Sample uids from dataset:")
        for i in range(3):
            print(self.dataframe[i].get(uid_key, self.dataframe[i].get("uid")), self.dataframe[i])

        # Save to permanent dataset folder instead of cache
        os.makedirs(self.dataset_dir, exist_ok=True)

        # Save as both HuggingFace dataset and parquet
        hf_save_dir = os.path.join(self.dataset_dir, "processed_with_uid_hf")
        parquet_save_path = os.path.join(self.dataset_dir, "processed_with_uid.parquet")

        print(f"Saving processed dataset with uid to {hf_save_dir}")
        self.dataframe.save_to_disk(hf_save_dir)

        print(f"Saving processed dataset as parquet to {parquet_save_path}")
        self.dataframe.to_parquet(parquet_save_path)

        # Also save to cache for compatibility
        cache_save_dir = os.path.join(self.cache_dir, "processed_with_uid")
        print(f"Also saving to cache: {cache_save_dir}")
        self.dataframe.save_to_disk(cache_save_dir)

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

            # Also save filtered dataset
            filtered_hf_dir = os.path.join(self.dataset_dir, "filtered_with_uid_hf")
            filtered_parquet_path = os.path.join(self.dataset_dir, "filtered_with_uid.parquet")

            print(f"Saving filtered dataset to {filtered_hf_dir}")
            self.dataframe.save_to_disk(filtered_hf_dir)

            print(f"Saving filtered dataset as parquet to {filtered_parquet_path}")
            self.dataframe.to_parquet(filtered_parquet_path)

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages_raw = example.pop(self.prompt_key)

        # Parse prompt if it's a string (convert to list)
        if isinstance(messages_raw, str):
            import ast
            try:
                messages: list = ast.literal_eval(messages_raw)
            except (ValueError, SyntaxError):
                # Fallback: treat as plain text message
                messages = [{"role": "user", "content": messages_raw}]
        else:
            messages: list = messages_raw

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # Parse extra_info if it's a string (convert to dictionary)
        if isinstance(row_dict.get("extra_info"), str):
            import ast

            try:
                row_dict["extra_info"] = ast.literal_eval(row_dict["extra_info"])
            except (ValueError, SyntaxError):
                row_dict["extra_info"] = {}
        elif row_dict.get("extra_info") is None:
            row_dict["extra_info"] = {}

        # Parse reward_model if it's a string (convert to dictionary)
        if isinstance(row_dict.get("reward_model"), str):
            import ast

            try:
                row_dict["reward_model"] = ast.literal_eval(row_dict["reward_model"])
            except (ValueError, SyntaxError):
                row_dict["reward_model"] = {}
        elif row_dict.get("reward_model") is None:
            row_dict["reward_model"] = {}

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs

        # --- Ensure the same uid is always used and never regenerated ---
        uid_key = self.universal_id_key
        # If neither uid_key nor "uid" is present, raise an error (should never happen if dataset is prepared correctly)
        if uid_key not in row_dict and "uid" not in row_dict:
            raise ValueError("uid missing from sample. This should not happen if dataset was prepared correctly.")
        # If only "uid" is present, copy it to uid_key
        if uid_key not in row_dict and "uid" in row_dict:
            row_dict[uid_key] = row_dict["uid"]
        # When logging or exporting, always use row_dict[uid_key] (never generate a new one)

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
