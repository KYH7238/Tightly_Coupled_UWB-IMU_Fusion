import rosbag
import rospy
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import os

bag_file = '../bag/all_filter.bag'
topics = ['/ekf', '/eskf', '/liekf', '/ukf', '/ground_truth']

data = {
    'ekf': {'positions': [], 'orientations': []},
    'eskf': {'positions': [], 'orientations': []},
    'liekf': {'positions': [], 'orientations': []},
    'ukf': {'positions': [], 'orientations': []},
    'gt':   {'positions': [], 'orientations': []}
}

with rosbag.Bag(bag_file) as bag:
    for topic, msg, t in bag.read_messages(topics=topics):
        key = topic.strip('/')
        pos = [msg.pose.position.x,
               msg.pose.position.y,
               msg.pose.position.z]
        quat = [msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w]
        euler = R.from_quat(quat).as_euler('xyz', degrees=True)

        if key == 'ground_truth':
            data['gt']['positions'].append(pos)
            data['gt']['orientations'].append(euler)
        else:
            data[key]['positions'].append(pos)
            data[key]['orientations'].append(euler)

def read_mppi(file_path):
    mppi = []
    with open(file_path, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split('\t')))
            mppi.append(vals)
    return mppi

mppi_file = '../config/mppi_pose1.txt'
if os.path.isfile(mppi_file):
    mppi_data = read_mppi(mppi_file)
    data['mppi'] = {
        'positions':    [d[:3] for d in mppi_data],
        'orientations': [list(np.degrees(d[3:6])) for d in mppi_data]
    }
else:
    print(f"[Warning] MPPI file not found: {mppi_file}")

def interpolate_data(seq, target_length):
    x = np.arange(len(seq))
    x_new = np.linspace(0, len(seq) - 1, target_length)
    arr = np.array(seq)
    return np.array([np.interp(x_new, x, arr[:, i]) for i in range(arr.shape[1])]).T

def calculate_rmse(gt, est):
    return np.sqrt(np.mean((gt - est)**2, axis=0))

def plot_positions_and_rpy(data):
    pos_keys = ['gt', 'ekf', 'eskf', 'liekf', 'ukf']
    if 'mppi' in data:
        pos_keys.append('mppi')

    max_len = max(len(data[k]['positions']) for k in pos_keys)
    interp_pos = {k: interpolate_data(data[k]['positions'], max_len) for k in pos_keys}

    for k in pos_keys:
        if k == 'gt': continue
        rmse = calculate_rmse(interp_pos['gt'], interp_pos[k])
        print(f"{k.upper():5s} Position RMSE (X,Y,Z): {rmse}")

    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 12))
    labels = {'gt':'GT','ekf':'EKF','eskf':'ESKF','liekf':'LIEKF','ukf':'UKF','mppi':'MPPI'}
    colors = {'gt':'black','ekf':'blue','eskf':'red','liekf':'green','ukf':'purple','mppi':'orange'}
    for idx, axis in enumerate(['X','Y','Z']):
        ax = axs1[idx]
        for k in pos_keys:
            ax.plot(interp_pos[k][:, idx],
                    label=labels[k],
                    color=colors[k],
                    linestyle='-' if k!='mppi' else '--')
        ax.set_ylabel(f"{axis} Position [m]")
        ax.grid()
        ax.legend(loc='upper right')
    axs1[2].set_xlabel('Samples')
    fig1.suptitle('Position Error')

    ori_keys = [k for k in pos_keys if data[k]['orientations'] is not None]
    if ori_keys:
        max_len_ori = max(len(data[k]['orientations']) for k in ori_keys)
        interp_ori = {k: interpolate_data(data[k]['orientations'], max_len_ori) for k in ori_keys}

        for k in ori_keys:
            if k == 'gt': continue
            rmse_ori = calculate_rmse(interp_ori['gt'], interp_ori[k])
            print(f"{k.upper():5s} Orientation RMSE (roll,pitch,yaw): {rmse_ori}")

        fig2, axs2 = plt.subplots(3, 1, figsize=(10, 12))
        for idx, angle in enumerate(['Roll','Pitch','Yaw']):
            ax = axs2[idx]
            for k in ori_keys:
                ax.plot(interp_ori[k][:, idx],
                        label=labels[k],
                        color=colors[k],
                        linestyle='-' if k!='mppi' else '--')
            ax.set_ylabel(f"{angle} [deg]")
            ax.grid()
            ax.legend(loc='upper right')
        axs2[2].set_xlabel('Samples')
        fig2.suptitle('Orientation Error')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_positions_and_rpy(data)
