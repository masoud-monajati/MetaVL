# MetaVL

This repo is adapted from MAGMA (https://github.com/Aleph-Alpha/magma)

In this repo, you can train MetaVL and do an inference on VQA-type data in few-shot setting.

# Phase one: meta-training LM:

For meta training, we followed https://github.com/facebookresearch/MetaICL repository to meta-train a small language model (GPT2-Medium) on MetaICL dataset. we ahve provided the model checkpoints in the checkpoint folder.

# Training

To train, please load the meta-trained LM in magma/language_model.py
then, run 

deepspeed ...


# Inference

To do an inference on any VQA data, you can run the inference_(VQA_type).py
