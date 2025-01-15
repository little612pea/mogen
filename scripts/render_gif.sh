 # absolute path to this project
 project_dir="/home/jovyan/mogen/motion-diffusion-model"
 # absolute path to blender app
 blender_app="/home/jovyan/mogen/ProgMoGen/progmogen/blender-2.93.18-linux-x64/blender"
 save_dir="mdm_finetune_initial_exp/mdm_finetune_actions_wo_physics/samples_mdm_finetune_actions_wo_phys"
 mesh_file="${project_dir}/save/${save_dir}"
 ${blender_app} --background --python render.py -- --cfg=./configs/render.yaml --dir=${mesh_file} --mode=video