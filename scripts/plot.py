import rosbag
import rospy
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

bag_file = '../bag/all_filter.bag'

topics = ['/ekf', '/eskf', '/liekf', '/ukf', '/ground_truth']

data = {
    'ekf': {'positions': [], 'orientations': []},
    'eskf': {'positions': [], 'orientations': []},
    'liekf': {'positions': [], 'orientations': []},
    'ukf': {'positions': [], 'orientations': []},
    'gt': {'positions': [], 'orientations': []}
}

bag = rosbag.Bag(bag_file)

for topic, msg, t in bag.read_messages(topics=topics):
    filter_type = topic.strip('/')
    position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    orientation_q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
    rotation = R.from_quat(orientation_q)
    euler = rotation.as_euler('xyz', degrees=True)
    if filter_type == 'ground_truth':
        data['gt']['positions'].append(position)
        data['gt']['orientations'].append(euler)
    else:
        data[filter_type]['positions'].append(position)
        data[filter_type]['orientations'].append(euler)

bag.close()

def interpolate_data(data, target_length):
    x = np.arange(len(data))
    x_new = np.linspace(0, len(data) - 1, target_length)
    data = np.array(data)
    data_interp = np.array([np.interp(x_new, x, data[:, i]) for i in range(data.shape[1])]).T
    return data_interp

def calculate_rmse(true_data, pred_data):
    mse = np.mean((true_data - pred_data) ** 2, axis=0)
    rmse = np.sqrt(mse)
    return rmse

def calculate_rmse_orientation(true_data_deg, pred_data_deg):
    true_data_rad = np.deg2rad(true_data_deg)
    pred_data_rad = np.deg2rad(pred_data_deg)
    angle_diff = np.unwrap(true_data_rad - pred_data_rad, axis=0)
    mse = np.mean(angle_diff ** 2, axis=0)
    rmse = np.sqrt(mse)
    return np.rad2deg(rmse)

def plot_positions(data):
    max_length = max(len(data['gt']['positions']), len(data['ekf']['positions']),
                     len(data['eskf']['positions']), len(data['liekf']['positions']), len(data['ukf']['positions']))

    gt_positions_interp = interpolate_data(data['gt']['positions'], max_length)
    ekf_positions_interp = interpolate_data(data['ekf']['positions'], max_length)
    eskf_positions_interp = interpolate_data(data['eskf']['positions'], max_length)
    liekf_positions_interp = interpolate_data(data['liekf']['positions'], max_length)
    ukf_positions_interp = interpolate_data(data['ukf']['positions'], max_length)

    rmse_ekf = calculate_rmse(gt_positions_interp, ekf_positions_interp)
    rmse_eskf = calculate_rmse(gt_positions_interp, eskf_positions_interp)
    rmse_liekf = calculate_rmse(gt_positions_interp, liekf_positions_interp)
    rmse_ukf = calculate_rmse(gt_positions_interp, ukf_positions_interp)

    print("EKF Position RMSE (X, Y, Z): ", rmse_ekf)
    print("ESKF Position RMSE (X, Y, Z): ", rmse_eskf)
    print("LIEKF Position RMSE (X, Y, Z): ", rmse_liekf)
    print("UKF Position RMSE (X, Y, Z): ", rmse_ukf)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(gt_positions_interp[:, 0], color='black', label='GT', linestyle='-')
    axs[0].plot(ekf_positions_interp[:, 0], color='blue', label='EKF', linestyle='-')
    axs[0].plot(eskf_positions_interp[:, 0], color='red', label='ESKF', linestyle='-')
    axs[0].plot(liekf_positions_interp[:, 0], color='green', label='LIEKF', linestyle='-')
    axs[0].plot(ukf_positions_interp[:, 0], color='purple', label='UKF', linestyle='-')

    axs[1].plot(gt_positions_interp[:, 1], color='black', label='GT', linestyle='-')
    axs[1].plot(ekf_positions_interp[:, 1], color='blue', label='EKF', linestyle='-')
    axs[1].plot(eskf_positions_interp[:, 1], color='red', label='ESKF', linestyle='-')
    axs[1].plot(liekf_positions_interp[:, 1], color='green', label='LIEKF', linestyle='-')
    axs[1].plot(ukf_positions_interp[:, 1], color='purple', label='UKF', linestyle='-')

    axs[2].plot(gt_positions_interp[:, 2], color='black', label='GT', linestyle='-')
    axs[2].plot(ekf_positions_interp[:, 2], color='blue', label='EKF', linestyle='-')
    axs[2].plot(eskf_positions_interp[:, 2], color='red', label='ESKF', linestyle='-')
    axs[2].plot(liekf_positions_interp[:, 2], color='green', label='LIEKF', linestyle='-')
    axs[2].plot(ukf_positions_interp[:, 2], color='purple', label='UKF', linestyle='-')

    axs[0].set_ylabel('X Position [m]')
    axs[1].set_ylabel('Y Position [m]')
    axs[2].set_ylabel('Z Position [m]')
    axs[2].set_xlabel('Samples')

    for ax in axs:
        ax.legend(loc='upper right')
        ax.grid()

    plt.tight_layout()
    plt.show()

def plot_orientations(data):
    max_length = max(len(data['gt']['orientations']), len(data['ekf']['orientations']),
                     len(data['eskf']['orientations']), len(data['liekf']['orientations']), len(data['ukf']['orientations']))

    gt_orientations_interp = interpolate_data(data['gt']['orientations'], max_length)
    ekf_orientations_interp = interpolate_data(data['ekf']['orientations'], max_length)
    eskf_orientations_interp = interpolate_data(data['eskf']['orientations'], max_length)
    liekf_orientations_interp = interpolate_data(data['liekf']['orientations'], max_length)
    ukf_orientations_interp = interpolate_data(data['ukf']['orientations'], max_length)

    rmse_ekf_ori = calculate_rmse_orientation(gt_orientations_interp, ekf_orientations_interp)
    rmse_eskf_ori = calculate_rmse_orientation(gt_orientations_interp, eskf_orientations_interp)
    rmse_liekf_ori = calculate_rmse_orientation(gt_orientations_interp, liekf_orientations_interp)
    rmse_ukf_ori = calculate_rmse_orientation(gt_orientations_interp, ukf_orientations_interp)

    print("EKF Orientation RMSE (Roll, Pitch, Yaw) in degrees: ", rmse_ekf_ori)
    print("ESKF Orientation RMSE (Roll, Pitch, Yaw) in degrees: ", rmse_eskf_ori)
    print("LIEKF Orientation RMSE (Roll, Pitch, Yaw) in degrees: ", rmse_liekf_ori)
    print("UKF Orientation RMSE (Roll, Pitch, Yaw) in degrees: ", rmse_ukf_ori)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(gt_orientations_interp[:, 0], color='black', label='GT', linestyle='-')
    axs[0].plot(ekf_orientations_interp[:, 0], color='blue', label='EKF', linestyle='-')
    axs[0].plot(eskf_orientations_interp[:, 0], color='red', label='ESKF', linestyle='-')
    axs[0].plot(liekf_orientations_interp[:, 0], color='green', label='LIEKF', linestyle='-')
    axs[0].plot(ukf_orientations_interp[:, 0], color='purple', label='UKF', linestyle='-')

    axs[1].plot(gt_orientations_interp[:, 1], color='black', label='GT', linestyle='-')
    axs[1].plot(ekf_orientations_interp[:, 1], color='blue', label='EKF', linestyle='-')
    axs[1].plot(eskf_orientations_interp[:, 1], color='red', label='ESKF', linestyle='-')
    axs[1].plot(liekf_orientations_interp[:, 1], color='green', label='LIEKF', linestyle='-')
    axs[1].plot(ukf_orientations_interp[:, 1], color='purple', label='UKF', linestyle='-')

    axs[2].plot(gt_orientations_interp[:, 2], color='black', label='GT', linestyle='-')
    axs[2].plot(ekf_orientations_interp[:, 2], color='blue', label='EKF', linestyle='-')
    axs[2].plot(eskf_orientations_interp[:, 2], color='red', label='ESKF', linestyle='-')
    axs[2].plot(liekf_orientations_interp[:, 2], color='green', label='LIEKF', linestyle='-')
    axs[2].plot(ukf_orientations_interp[:, 2], color='purple', label='UKF', linestyle='-')

    axs[0].set_ylabel('Roll [deg]')
    axs[1].set_ylabel('Pitch [deg]')
    axs[2].set_ylabel('Yaw [deg]')
    axs[2].set_xlabel('Samples')

    for ax in axs:
        ax.legend(loc='upper right')
        ax.grid()

    plt.tight_layout()
    plt.show()

plot_positions(data)
plot_orientations(data)
