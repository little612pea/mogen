python -m eval.eval_humanml \
        --model_path ./save/01_15_linear/wo_physics_then_all_text/model000580000.pt  \
        --eval_mode debug \
        --arch_decoupling="linear" \
        --stage="full-text"