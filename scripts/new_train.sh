#用原始模型测试
python -m train.train_mdm \
        --resume_checkpoint save/mdm_finetune_linear_all_texts_01_15_continue_1935/model000550000.pt \
        --save_dir save/mdm_finetune_linear_all_texts_then_only_actions_then_wo_physics_01_15_continue_2214 \
        --save_interval 5000 \
        --lr 5e-5 \
        --stage "wo-physics" \
        --arch_decoupling="linear"
        
        
        
# texts stage 0: full stage 1: only-actions stage 2: remove-physics
#命名规范：mdm_finetune_{训练阶段}_{模型结构}_{日期}
python -m train.train_mdm \
        --resume_checkpoint save/mdm_finetune_till_converge_2025_01_13_test_only_actions_first_2/model000500000.pt \
        --save_dir save/mdm_finetune_till_converge_2025_01_13_temp \
        --save_interval 5000 \
        --stage "full-text" \
        --lr 5e-5 \
        --arch_decoupling="linear"

python -m train.train_mdm \
        --resume_checkpoint save/mdm_finetune_linear_full_text_multi_head_comp_1126/model000505000.pt \
        --save_dir save/mdm_finetune_linear_only_actions_multi_head_comp_1127 \
        --save_interval 5000 \
        --lr 5e-5 \
        --stage "only-actions" \
        --arch_decoupling="multi_head_comp" \
        --weight_decay 0.0001 \
        --eval_during_training 

python -m train.train_mdm \
        --resume_checkpoint ./save/mdm_finetune_till_converge_2025_01_13_then_wo_physics_2/model000525000.pt \
        --save_dir save/mdm_finetune_initial_exp/mdm_finetune_full_01_16 \
        --save_interval 5000 \
        --lr 5e-5 \
        --stage "full-text" \
        --arch_decoupling="linear" \
        --weight_decay 0.0001
        
python -m eval.eval_humanml \
        --model_path ./save/multi-head-comp-1127-tryout/stage3_wo_physics/model000635000.pt  \
        --eval_mode debug \
        --arch_decoupling="multi_head_comp" \
        --stage="full-text"

python -m eval.eval_humanml \
        --model_path ./save/01_15_linear/all_text_1/model000530000.pt  \
        --eval_mode debug \
        --arch_decoupling="linear" \
        --stage="full-text"
#generate阶段也需提供stage,属于必须的参数
python -m sample.generate \
        --model_path ./save/multi-head-comp-1127-tryout/stage3_wo_physics/model000635000.pt \
        --text_prompt "the person walked forward and is picking up his toolbox."  \
        --arch_decoupling="multi_head_comp" --stage="full-text"
        
python -m sample.generate \
        --model_path ./save/01_15_all_text_linear_1/model000530000.pt \
        --text_prompt "a man faces the camera and stands in place while raising his left arm, then waves his left hand before returning to a standing position."  \
        --arch_decoupling="linear" --stage="full-text"

       