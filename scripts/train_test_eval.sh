#!/bin/bash

# 检查是否提供了模型路径和保存路径
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <resume_checkpoint_path> <save_dir_path>"
    exit 1
fi

# 提取输入的参数
RESUME_CHECKPOINT_1="./save/humanml_trans_enc_512/model000475000.pt"
SAVE_DIR_1="./save/actions_conditioned_stage1/1120-1"

RESUME_CHECKPOINT_2="./save/actions_conditioned_stage1/1120-1"
SAVE_DIR_2="./save/actions_conditioned_stage2/1120-1"




cd ../HumanML3D
cp ./texts_humanml/* ./texts
cd ../motion-diffusion-model
# only texts阶段
echo "Starting training from checkpoint $RESUME_CHECKPOINT_1..."
python -m train.train_mdm --resume_checkpoint "$RESUME_CHECKPOINT_1" --save_dir "$SAVE_DIR_1" --save_interval 5000

cd ../HumanML3D
cp ./texts_remove_physics_processed/* ./texts
cd ../motion-diffusion-model
echo "Starting training from checkpoint $RESUME_CHECKPOINT_2..."
python -m train.train_mdm --resume_checkpoint "$RESUME_CHECKPOINT_2" --save_dir "$SAVE_DIR_2" --save_interval 5000

SAVE_DIR_2="./save/mdm_finetune_1119"
# Sample阶段
echo "Generating samples using the model from $SAVE_DIR_2..."
LATEST_MODEL=$(find "$SAVE_DIR_2" -type f -name "model*.pt" | sort -V | tail -n 1)
SAMPLE_MODEL="$SAVE_DIR_2/$(basename $LATEST_MODEL)"
python -m sample.generate --model_path "$SAMPLE_MODEL" --text_prompt "the person walked forward and is picking up his toolbox."

# Eval阶段
echo "Evaluating the model from $SAVE_DIR_2..."
python -m eval.eval_humanml --model_path "$SAMPLE_MODEL"

# 打印完成消息
echo "Training, Sampling, and Evaluation for $RESUME_CHECKPOINT_2 are completed."
