import argparse
import torch
import lmms_eval
from lmms_eval.tasks import TaskManager
from lmms_eval.evaluator import simple_evaluate

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import time

model_path = "/code/yingqi/models/liuhaotian/llava-v1.5-7b"
model_base = None
device = "cuda"
load_8bit = False
load_4bit = False

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit,load_4bit, device=device, cache_dir="/code/yingqi/models")

# instantiate an LM subclass that takes your initialized model and can run
# - `Your_LMM.loglikelihood()`
# - `Your_LMM.generate_until()`
# lmm_obj = Your_LMM(model=model, batch_size=16)

# indexes all tasks from the `lmms_eval/tasks` subdirectory.
# Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
# to include a set of tasks in a separate directory.
task_manager = TaskManager()

# Setting `task_manager` to the one above is optional and should generally be done
# if you want to include tasks from paths other than ones in `lmms_eval/tasks`.
# `simple_evaluate` will instantiate its own task_manager if it is set to None here.
results = simple_evaluate( # call simple_evaluate
    model="llava",
    model_args=  "/code/yingqi/models/liuhaotian/llava-v1.5-7b, conv_template=vicuna_v1, temperature=0",
    tasks=["gqa"],
    num_fewshot=0,
    task_manager=task_manager,
    batch_size=1,
    use_cache=True,
)