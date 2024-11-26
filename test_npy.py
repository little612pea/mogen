import numpy as np

# 加载 .npy 文件
file_path = './save/mdm_finetune_initial_exp/mdm_finetune_actions_wo_physics/samples_mdm_finetune_actions_wo_phys/sample00_rep00_smpl_params.npy'
data = np.load(file_path, allow_pickle=True)

# 打印数据内容
print(data)