import open3d as o3d
import os
import cv2
import numpy as np

# 输入和输出路径
obj_folder = '../save/mdm_finetune_initial_exp/mdm_finetune_actions_wo_physics/samples_mdm_finetune_actions_wo_phys/sample00_rep00_obj'  # 存储 obj 文件的文件夹
output_folder = 'rendered_frames'  # 用于保存每一帧渲染图像的文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 设置视频参数
video_name = 'output_video.mp4'
fps = 30  # 每秒帧数

# 获取 obj 文件列表
obj_files = [f for f in os.listdir(obj_folder) if f.endswith('.obj')]

# 渲染每个 obj 文件
for i, obj_file in enumerate(sorted(obj_files)):
    obj_path = os.path.join(obj_folder, obj_file)

    # 读取 obj 文件
    mesh = o3d.io.read_triangle_mesh(obj_path)

    # 创建一个可视化窗口，不显示窗口但进行渲染
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    # 设置视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])  # 设置视角方向
    ctr.set_lookat([0, 0, 0])  # 设置视角焦点
    ctr.set_up([0, -1, 0])     # 设置上方向

    # 添加光照
    light = o3d.visualization.Light()
    light.ambient_intensity = 0.5
    light.diffuse_intensity = 0.8
    light.specular_intensity = 1.0
    light.attenuation = [0.0, 0.0, 0.0]
    light.position = [0, 0, 5]
    vis.add_light(light)

    # 渲染到图像
    image = vis.capture_screen_float_buffer(do_render=True)
    image = (255 * np.asarray(image)).astype(np.uint8)  # 转换为8位图像
    output_path = os.path.join(output_folder, f'frame_{i:04d}.png')
    cv2.imwrite(output_path, image)

    # 清理
    vis.destroy_window()

print("所有 OBJ 文件已渲染为图像帧。")

import cv2
import os

# 渲染帧所在文件夹
image_folder = 'rendered_frames'
video_name = 'output_video.mp4'

# 获取所有图像帧文件
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 创建视频文件
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# 将每一帧写入视频
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# 完成视频写入并释放
cv2.destroyAllWindows()
video.release()

print(f"视频 {video_name} 生成成功。")
