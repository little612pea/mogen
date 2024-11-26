#用原始模型测试
python -m train.train_mdm \
        --resume_checkpoint save/humanml_trans_enc_512/model000475000.pt \
        --save_dir save/mdm_finetune_linear_all_texts_1126 \
        --save_interval 5000 \
        --lr 5e-5 \
        --stage 3 \
        --arch_decoupling="none"
# texts stage 0: full stage 1: only-actions stage 2: remove-physics
#命名规范：mdm_finetune_{训练阶段}_{模型结构}_{日期}
python -m train.train_mdm \
        --resume_checkpoint save/humanml_trans_enc_512/model000475000.pt \
        --save_dir save/mdm_finetune_linear_only_actions_linear_1126 \
        --save_interval 5000 \
        --lr 5e-5 \
        --stage "only-actions" \
        --arch_decoupling="linear"
python -m eval.eval_humanml \
        --model_path ./save/mdm_finetune_initial_exp/mdm_finetune_actions_wo_physics/model000690000.pt \
        --eval_mode debug \
        --arch_decoupling="initial_exp" 
python -m sample.generate \
        --model_path ./save/mdm_finetune_initial_exp/mdm_finetune_actions_wo_physics/model000690000.pt \
        --text_prompt "the person walked forward and is picking up his toolbox." \
        --arch_decoupling="initial_exp" 