
import sys
import json
import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.camera as pc
import pytransform3d.visualizer as pv
import pytransform3d.plot_utils as pu
import matplotlib.pyplot as plt

# mesh_filename = 'logs/blendswap_whitehouse_origin_v11_near1_far10_path/blendswap_whitehouse_origin_v11_near1_far10_path_256.obj' # sys.argv[-2]
camera_filename = 'data/blendswap_whitehouse_origin_v11/transforms_train_v11apple_table.json' # sys.argv[-1]
# camera_filename = 'data/blendswap_whitehouse_origin_v11/transforms_train_path.json' # sys.argv[-1]

with open(camera_filename, "r") as f:
    json_train = json.load(f)

camera_poses = [cur_dict['transform_matrix'] for cur_dict in json_train["frames"]]
# camera_poses = 
# camera_intrinsics = cameras["intrinsics"][0]

px_focal_length = float(json_train['fl_x']) # float(camera_intrinsics["pxFocalLength"])
px_principal_point_x = float(json_train["cx"]) # float(camera_intrinsics["principalPoint"][0])
px_principal_point_y = float(json_train["cy"]) # float(camera_intrinsics["principalPoint"][1])
M = np.array([
    [px_focal_length, 0, px_principal_point_x],
    [0, px_focal_length, px_principal_point_y],
    [0, 0, 1]
])
sensor_size = (float(json_train["w"]), float(json_train["h"]))

transformation_matrices = np.empty((len(camera_poses), 4, 4))
for i, camera_pose in enumerate(camera_poses):
    # R = np.array(list(map(float, camera_pose["pose"]["transform"]["rotation"]))).reshape(3, 3)
    # p = np.array(list(map(float, camera_pose["pose"]["transform"]["center"])))
    # transformation_matrices[i] = pt.transform_from(R=R, p=p)
    
    transformation_matrices[i] = np.array(camera_pose)
    transformation_matrices[i][:3, 3] = transformation_matrices[i][:3, 3] * 0.3
    
    transformation_matrices[i][0, 2] = -transformation_matrices[i][0, 2]
    transformation_matrices[i][1, 2] = -transformation_matrices[i][1, 2]
    transformation_matrices[i][2, 0] = -transformation_matrices[i][2, 0]
    transformation_matrices[i][2, 1] = -transformation_matrices[i][2, 1]

    transformation_matrices[i][2, 3] = -transformation_matrices[i][2, 3]

fig = plt.figure(figsize=(20, 20))
# ax = pu.plot_mesh(mesh_filename, s=5 * np.ones(3), alpha=0.3)

# colors = cycle("b")
for i, pose in enumerate(transformation_matrices):
    # pt.plot_transform(ax=ax, A2B=pose, s=0.1)
    pc.plot_camera(M=M, cam2world=pose, virtual_image_distance=0.1, sensor_size=sensor_size, c='b')
# fig.show()


# pos_min = np.min(transformation_matrices[:, :3, 3], axis=0)
# pos_max = np.max(transformation_matrices[:, :3, 3], axis=0)
# center = (pos_max + pos_min) / 2.0
# max_half_extent = max(pos_max - pos_min) / 2.0
# ax.set_xlim((center[0] - max_half_extent, center[0] + max_half_extent))
# ax.set_ylim((center[1] - max_half_extent, center[1] + max_half_extent))
# ax.set_zlim((center[2] - max_half_extent, center[2] + max_half_extent))

# ax.view_init(azim=110, elev=40)

# plt.xlim()
# plt.xlim((center[0] - max_half_extent, center[0] + max_half_extent))
# plt.ylim((center[1] - max_half_extent, center[1] + max_half_extent))
# plt.zlim((center[2] - max_half_extent, center[2] + max_half_extent))

fig.savefig('vis_camera_' + camera_filename.split('/')[-1].split('.')[0] + '.jpg')