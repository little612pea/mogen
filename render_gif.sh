# absolute path to this project
project_dir="/home/jovyan/mogen/motion-diffusion-model"
# absolute path to blender app
blender_app="/home/jovyan/mogen/ProgMoGen/progmogen/blender-2.93.18-linux-x64/blender"
save_dir="mdm_finetune_initial_exp/mdm_finetune_actions_wo_physics/samples_mdm_finetune_actions_wo_phys"
mesh_file="${project_dir}/save/${save_dir}/sample00_rep00_smpl_params.npy"
# blender_app="/home/jovyan/mogen/ProgMoGen/progmogen/blender-2.93.18-linux-x64/blender"
# save_dir="mdm_finetune_initial_exp/mdm_finetune_actions_wo_physics"
# mesh_file="${project_dir}/save/${save_dir}/gen1_smpl_params.npy"
echo ${mesh_file}
cd ./TEMOS-master
${blender_app} --background --python render.py -- npy=${mesh_file} canonicalize=true mode="sequence"
${blender_app} --background --python render.py -- npy=${mesh_file} canonicalize=true mode="video"
echo "[Results are saved in ${save_fig_dir}/gen_smpl]"
cd -