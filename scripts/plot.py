# import rosbag
# import rospy
# import numpy as np
# import matplotlib.pyplot as plt
# from geometry_msgs.msg import PoseStamped
# from scipy.spatial.transform import Rotation as R
# from scipy.interpolate import interp1d
    
# bag_file = '../bag/all_filter.bag'

# topics = ['/ekf', '/eskf', '/liekf', '/ukf', '/ground_truth']

# data = {
#     'ekf': {'positions': [], 'orientations': []},
#     'eskf': {'positions': [], 'orientations': []},
#     'liekf': {'positions': [], 'orientations': []},
#     'ukf': {'positions': [], 'orientations': []},
#     'gt': {'positions': [], 'orientations': []}
# }

# bag = rosbag.Bag(bag_file)

# for topic, msg, t in bag.read_messages(topics=topics):
#     filter_type = topic.strip('/')
#     position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
#     orientation_q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
#     rotation = R.from_quat(orientation_q)
#     euler = rotation.as_euler('xyz', degrees=True)
#     if filter_type == 'ground_truth':
#         data['gt']['positions'].append(position)
#         data['gt']['orientations'].append(euler)
#     else:
#         data[filter_type]['positions'].append(position)
#         data[filter_type]['orientations'].append(euler)

# bag.close()

# def interpolate_data(data, target_length):
#     x = np.arange(len(data))
#     x_new = np.linspace(0, len(data) - 1, target_length)
#     data = np.array(data)
#     data_interp = np.array([np.interp(x_new, x, data[:, i]) for i in range(data.shape[1])]).T
#     return data_interp

# def calculate_rmse(true_data, pred_data):
#     mse = np.mean((true_data - pred_data) ** 2, axis=0)
#     rmse = np.sqrt(mse)
#     return rmse

# def calculate_rmse_orientation(true_data_deg, pred_data_deg):
#     true_data_rad = np.deg2rad(true_data_deg)
#     pred_data_rad = np.deg2rad(pred_data_deg)
#     angle_diff = np.unwrap(true_data_rad - pred_data_rad, axis=0)
#     mse = np.mean(angle_diff ** 2, axis=0)
#     rmse = np.sqrt(mse)
#     return np.rad2deg(rmse)

# def plot_positions(data):
#     max_length = max(len(data['gt']['positions']), len(data['ekf']['positions']),
#                      len(data['eskf']['positions']), len(data['liekf']['positions']), len(data['ukf']['positions']))

#     gt_positions_interp = interpolate_data(data['gt']['positions'], max_length)
#     ekf_positions_interp = interpolate_data(data['ekf']['positions'], max_length)
#     eskf_positions_interp = interpolate_data(data['eskf']['positions'], max_length)
#     liekf_positions_interp = interpolate_data(data['liekf']['positions'], max_length)
#     ukf_positions_interp = interpolate_data(data['ukf']['positions'], max_length)

#     rmse_ekf = calculate_rmse(gt_positions_interp, ekf_positions_interp)
#     rmse_eskf = calculate_rmse(gt_positions_interp, eskf_positions_interp)
#     rmse_liekf = calculate_rmse(gt_positions_interp, liekf_positions_interp)
#     rmse_ukf = calculate_rmse(gt_positions_interp, ukf_positions_interp)

#     print("EKF Position RMSE (X, Y, Z): ", rmse_ekf)
#     print("ESKF Position RMSE (X, Y, Z): ", rmse_eskf)
#     print("LIEKF Position RMSE (X, Y, Z): ", rmse_liekf)
#     print("UKF Position RMSE (X, Y, Z): ", rmse_ukf)

#     fig, axs = plt.subplots(3, 1, figsize=(10, 12))

#     axs[0].plot(gt_positions_interp[:, 0], color='black', label='GT', linestyle='-')
#     axs[0].plot(ekf_positions_interp[:, 0], color='blue', label='EKF', linestyle='-')
#     axs[0].plot(eskf_positions_interp[:, 0], color='red', label='ESKF', linestyle='-')
#     axs[0].plot(liekf_positions_interp[:, 0], color='green', label='LIEKF', linestyle='-')
#     axs[0].plot(ukf_positions_interp[:, 0], color='purple', label='UKF', linestyle='-')

#     axs[1].plot(gt_positions_interp[:, 1], color='black', label='GT', linestyle='-')
#     axs[1].plot(ekf_positions_interp[:, 1], color='blue', label='EKF', linestyle='-')
#     axs[1].plot(eskf_positions_interp[:, 1], color='red', label='ESKF', linestyle='-')
#     axs[1].plot(liekf_positions_interp[:, 1], color='green', label='LIEKF', linestyle='-')
#     axs[1].plot(ukf_positions_interp[:, 1], color='purple', label='UKF', linestyle='-')

#     axs[2].plot(gt_positions_interp[:, 2], color='black', label='GT', linestyle='-')
#     axs[2].plot(ekf_positions_interp[:, 2], color='blue', label='EKF', linestyle='-')
#     axs[2].plot(eskf_positions_interp[:, 2], color='red', label='ESKF', linestyle='-')
#     axs[2].plot(liekf_positions_interp[:, 2], color='green', label='LIEKF', linestyle='-')
#     axs[2].plot(ukf_positions_interp[:, 2], color='purple', label='UKF', linestyle='-')

#     axs[0].set_ylabel('X Position [m]')
#     axs[1].set_ylabel('Y Position [m]')
#     axs[2].set_ylabel('Z Position [m]')
#     axs[2].set_xlabel('Samples')

#     for ax in axs:
#         ax.legend(loc='upper right')
#         ax.grid()

#     plt.tight_layout()
#     plt.show()

# def plot_orientations(data):
#     max_length = max(len(data['gt']['orientations']), len(data['ekf']['orientations']),
#                      len(data['eskf']['orientations']), len(data['liekf']['orientations']), len(data['ukf']['orientations']))

#     gt_orientations_interp = interpolate_data(data['gt']['orientations'], max_length)
#     ekf_orientations_interp = interpolate_data(data['ekf']['orientations'], max_length)
#     eskf_orientations_interp = interpolate_data(data['eskf']['orientations'], max_length)
#     liekf_orientations_interp = interpolate_data(data['liekf']['orientations'], max_length)
#     ukf_orientations_interp = interpolate_data(data['ukf']['orientations'], max_length)

#     rmse_ekf_ori = calculate_rmse_orientation(gt_orientations_interp, ekf_orientations_interp)
#     rmse_eskf_ori = calculate_rmse_orientation(gt_orientations_interp, eskf_orientations_interp)
#     rmse_liekf_ori = calculate_rmse_orientation(gt_orientations_interp, liekf_orientations_interp)
#     rmse_ukf_ori = calculate_rmse_orientation(gt_orientations_interp, ukf_orientations_interp)

#     print("EKF Orientation RMSE (Roll, Pitch, Yaw) in degrees: ", rmse_ekf_ori)
#     print("ESKF Orientation RMSE (Roll, Pitch, Yaw) in degrees: ", rmse_eskf_ori)
#     print("LIEKF Orientation RMSE (Roll, Pitch, Yaw) in degrees: ", rmse_liekf_ori)
#     print("UKF Orientation RMSE (Roll, Pitch, Yaw) in degrees: ", rmse_ukf_ori)

#     fig, axs = plt.subplots(3, 1, figsize=(10, 12))

#     axs[0].plot(gt_orientations_interp[:, 0], color='black', label='GT', linestyle='-')
#     axs[0].plot(ekf_orientations_interp[:, 0], color='blue', label='EKF', linestyle='-')
#     axs[0].plot(eskf_orientations_interp[:, 0], color='red', label='ESKF', linestyle='-')
#     axs[0].plot(liekf_orientations_interp[:, 0], color='green', label='LIEKF', linestyle='-')
#     axs[0].plot(ukf_orientations_interp[:, 0], color='purple', label='UKF', linestyle='-')

#     axs[1].plot(gt_orientations_interp[:, 1], color='black', label='GT', linestyle='-')
#     axs[1].plot(ekf_orientations_interp[:, 1], color='blue', label='EKF', linestyle='-')
#     axs[1].plot(eskf_orientations_interp[:, 1], color='red', label='ESKF', linestyle='-')
#     axs[1].plot(liekf_orientations_interp[:, 1], color='green', label='LIEKF', linestyle='-')
#     axs[1].plot(ukf_orientations_interp[:, 1], color='purple', label='UKF', linestyle='-')

#     axs[2].plot(gt_orientations_interp[:, 2], color='black', label='GT', linestyle='-')
#     axs[2].plot(ekf_orientations_interp[:, 2], color='blue', label='EKF', linestyle='-')
#     axs[2].plot(eskf_orientations_interp[:, 2], color='red', label='ESKF', linestyle='-')
#     axs[2].plot(liekf_orientations_interp[:, 2], color='green', label='LIEKF', linestyle='-')
#     axs[2].plot(ukf_orientations_interp[:, 2], color='purple', label='UKF', linestyle='-')

#     axs[0].set_ylabel('Roll [deg]')
#     axs[1].set_ylabel('Pitch [deg]')
#     axs[2].set_ylabel('Yaw [deg]')
#     axs[2].set_xlabel('Samples')

#     for ax in axs:
#         ax.legend(loc='upper right')
#         ax.grid()

#     plt.tight_layout()
#     plt.show()

# plot_positions(data)
# plot_orientations(data)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d
plt.rcParams.update({
    "font.size": 16,         # 기본 폰트 크기
    "axes.labelsize": 18,    # x, y 라벨 크기
    "xtick.labelsize": 14,   # x축 눈금 크기
    "ytick.labelsize": 14,   # y축 눈금 크기
    "legend.fontsize": 18,   # 범례 크기
    "axes.titlesize": 18     # 제목 크기
})
bag_file = '../bag/tightly_all_filter.bag'
gt_file  = '../config/0612_hw3_gt.txt'

topics = ['/ekf', '/eskf', '/liekf', '/ukf'] 
name_map = {'/ekf':'ekf','/eskf':'eskf','/liekf':'liekf','/ukf':'ukf'}

DT = 0.01 
POS_SCALE = 0.001
X_OFFSET = 4.4
Y_OFFSET = 4.0

streams = {k: {'t': [], 'pos': [], 'quat': []}
           for k in ['ekf','eskf','liekf','ukf']}

bag = rosbag.Bag(bag_file)
for topic, msg, t in bag.read_messages(topics=topics):
    key = name_map[topic]
    ts = msg.header.stamp.to_sec() if hasattr(msg, 'header') else t.to_sec()
    p = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
    streams[key]['t'].append(ts)
    streams[key]['pos'].append(p)
    streams[key]['quat'].append(q)
bag.close()

for k in streams:
    t = np.asarray(streams[k]['t'])
    order = np.argsort(t)
    streams[k]['t']   = t[order]
    streams[k]['pos'] = np.asarray(streams[k]['pos'], dtype=float)[order]
    streams[k]['quat']= np.asarray(streams[k]['quat'], dtype=float)[order]

def load_gt_txt_pos_mm_rpy_deg(path):
    pos_list = []
    quat_list = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            vals = [float(v) for v in line.strip().split()]
            if len(vals) < 6:
                continue
            x_m = vals[0] * POS_SCALE + X_OFFSET
            y_m = vals[1] * POS_SCALE + Y_OFFSET
            z_m = vals[2] * POS_SCALE - 0.1
            pos = np.array([x_m, y_m, z_m], dtype=float)

            r_deg, p_deg, y_deg = vals[3], vals[4], vals[5]
            y_deg = -y_deg  
            quat = R.from_euler('xyz', [r_deg, p_deg, y_deg], degrees=True).as_quat()

            pos_list.append(pos)
            quat_list.append(quat)

    pos = np.asarray(pos_list, dtype=float)
    quat = np.asarray(quat_list, dtype=float)
    return pos, quat


gt_pos_raw, gt_quat_raw = load_gt_txt_pos_mm_rpy_deg(gt_file)

t_start = max(streams[s]['t'][0] for s in streams if len(streams[s]['t'])>0)
t_end   = min(streams[s]['t'][-1] for s in streams if len(streams[s]['t'])>0)
if t_end <= t_start:
    raise RuntimeError("공통 시간 구간이 없습니다.")
t_grid = np.arange(t_start, t_end, DT)

gt_t_raw = np.linspace(t_start, t_end, gt_pos_raw.shape[0])

def resample_pos(t, pos, t_grid):
    f = [interp1d(t, pos[:,i], kind='linear', bounds_error=False, fill_value=np.nan)
         for i in range(3)]
    return np.vstack([fi(t_grid) for fi in f]).T 

def resample_quat(t, quat, t_grid):
    out = np.full((t_grid.size, 4), np.nan)
    inside = (t_grid >= t[0]) & (t_grid <= t[-1])
    if inside.any():
        slerp = Slerp(t, R.from_quat(quat))
        out[inside] = slerp(t_grid[inside]).as_quat()
    return out

res = {'gt': {
    'pos':  resample_pos(gt_t_raw,  gt_pos_raw,  t_grid),
    'quat': resample_quat(gt_t_raw, gt_quat_raw, t_grid),
}}

for k in streams:
    res[k] = {
        'pos':  resample_pos(streams[k]['t'],  streams[k]['pos'],  t_grid),
        'quat': resample_quat(streams[k]['t'], streams[k]['quat'], t_grid),
    }

def common_mask(a, b):
    return (~np.isnan(a).any(axis=1)) & (~np.isnan(b).any(axis=1))

def rmse_pos_per_axis(gt_pos, est_pos):
    m = common_mask(gt_pos, est_pos)
    diff = gt_pos[m] - est_pos[m]
    return np.sqrt(np.mean(diff**2, axis=0)) 

def rmse_euler_per_axis_deg(gt_quat, est_quat, seq='xyz'):
    m = common_mask(gt_quat, est_quat)
    Rt = R.from_quat(gt_quat[m]); Re = R.from_quat(est_quat[m])
    Et = Rt.as_euler(seq, degrees=True); Ee = Re.as_euler(seq, degrees=True)

    def unwrap_deg(arr):
        return np.degrees(np.unwrap(np.radians(arr), axis=0))

    Et_u = unwrap_deg(Et)
    Ee_u = unwrap_deg(Ee)
    return np.sqrt(np.mean((Et_u - Ee_u)**2, axis=0))  

for k in ['ekf','eskf','liekf','ukf']:
    r_pos = rmse_pos_per_axis(res['gt']['pos'], res[k]['pos'])
    r_ori = rmse_euler_per_axis_deg(res['gt']['quat'], res[k]['quat'], seq='xyz')
    print(f"{k.upper()} Position RMSE (X, Y, Z): {r_pos}")
    print(f"{k.upper()} Orientation RMSE (Roll, Pitch, Yaw) in degrees: {r_ori}")

x = np.arange(t_grid.size)
# labels = [('gt','black'), ('ekf','blue'), ('eskf','red'), ('liekf','green'), ('ukf','purple')]
labels = [('gt','black'), ('ekf','blue'), ('eskf','red'), ('ukf','purple')]

fig_p, axs_p = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
pos_names = ['X Position [m]', 'Y Position [m]', 'Z Position [m]']
for i in range(3):
    for name, color in labels:
        y = res[name]['pos'][:, i]
        m = ~np.isnan(y)
        axs_p[i].plot(x[m], y[m], color=color, label=name.upper() if i==0 else None)
    axs_p[i].set_ylabel(pos_names[i]); axs_p[i].grid(True)
axs_p[-1].set_xlabel('Samples')
axs_p[0].legend(loc='best')
fig_p.tight_layout()

def euler_xyz_from_quat(q):
    E = R.from_quat(q).as_euler('xyz', degrees=True)
    E_u = np.column_stack([
        np.degrees(np.unwrap(np.radians(E[:,0]))),
        np.degrees(np.unwrap(np.radians(E[:,1]))),
        np.degrees(np.unwrap(np.radians(E[:,2]))),
    ])
    return E_u

fig_o, axs_o = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
onames = ['Roll [deg]', 'Pitch [deg]', 'Yaw [deg]']
for name, color in labels:
    E = euler_xyz_from_quat(res[name]['quat'])
    for i in range(3):
        y = E[:, i]; m = ~np.isnan(y)
        axs_o[i].plot(x[m], y[m], color=color, label=name.upper() if i==0 else None)
for i in range(3):
    axs_o[i].set_ylabel(onames[i]); axs_o[i].grid(True)
axs_o[-1].set_xlabel('Samples')
axs_o[0].legend(loc='best')
fig_o.tight_layout()

plt.show()
