import numpy as np
import rospy
from nlink_parser.msg import LinktrackTagframe0
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

class SE3IEKF:
    def __init__(self):
        self.state = {
            "p": np.array([4.36, 4, 0]),
            "R": np.eye(3),
            "v": np.zeros(3),
            "b_acc": np.zeros(3),
            "b_gyro": np.zeros(3)
        }
        self.m_matP = np.zeros((15, 15), dtype=np.float32)
        self.m_matQ = 0.001 * np.eye(15, dtype=np.float64)
        self.m_matQ[9:12, 9:12] = 0.0015387262937311438 * np.eye(3)
        self.m_matQ[12:15, 12:15] = 1.0966208586777776e-06 * np.eye(3)
        self.m_matR = 0.08 * np.eye(8, dtype=np.float64)
        self.m_vecZ = np.zeros(8, dtype=np.float64)
        self.m_vech = np.zeros(8, dtype=np.float64)
        self.uwb_position = np.array([
            [0, 0, 8.86, 8.86, 0, 0, 8.86, 8.86],
            [0, 8.00, 8.00, 0, 0, 8.00, 8.00, 0],
            [0.2, 0.2, 0.2, 0.2, 2.40, 2.40, 2.40, 2.40]
        ], dtype=np.float64)
        rospy.Subscriber("/nlink_linktrack_tagframe0", LinktrackTagframe0, self.uwb_callback)
        rospy.Subscriber("/imu/data", Imu, self.imu_callback)
        self.pub_ekf = rospy.Publisher("/result_iekf", PoseStamped, queue_size=10)
        self.delta_t = 0
        self.before_t = None
        self.uwb_init = False
        self.imu_init = False
        self.imu_data_queue = []
        self.uwb_data_queue = []

    def uwb_callback(self, msg):
        uwb_time = msg.system_time / 1e3
        if not self.uwb_init:
            self.uwb_init_time = uwb_time
            self.uwb_init = True
        uwb_data = {
            "timestamp": uwb_time - self.uwb_init_time,
            "pos_3d": [msg.pos_3d[0], msg.pos_3d[1], msg.pos_3d[2]],
            "dis_arr": [msg.dis_arr[i] for i in range(8)]
        }
        self.uwb_data_queue.append(uwb_data)
        self.process_data()

    def imu_callback(self, msg):
        imu_time = msg.header.stamp.to_sec()
        if not self.imu_init:
            self.imu_init_time = imu_time
            self.imu_init = True
        imu_data = {
            "timestamp": imu_time - self.imu_init_time,
            "linear_acc": [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
            "angular_vel": [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        }
        self.imu_data_queue.append(imu_data)
        self.process_data()

    def process_data(self):
        if len(self.imu_data_queue) > 1 and len(self.uwb_data_queue) > 0:
            imu_data_1 = self.imu_data_queue[0]
            imu_data_2 = self.imu_data_queue[-1]
            uwb_data = self.uwb_data_queue[0]
            if imu_data_1["timestamp"] <= uwb_data["timestamp"] <= imu_data_2["timestamp"]:
                t1 = imu_data_1["timestamp"]
                t2 = imu_data_2["timestamp"]
                t = uwb_data["timestamp"]

                alpha = (t - t1) / (t2 - t1)

                linear_acc = (1 - alpha) * np.array(imu_data_1["linear_acc"]) + alpha * np.array(imu_data_2["linear_acc"])
                angular_vel = (1 - alpha) * np.array(imu_data_1["angular_vel"]) + alpha * np.array(imu_data_2["angular_vel"])

                self.delta_t = t - self.before_t if self.before_t else 0.01
                self.before_t = t
                self.linear_acc = linear_acc
                self.angular_vel = angular_vel
                self.m_vecZ = np.array(uwb_data["dis_arr"])

                self.prediction()
                self.correction()

                self.uwb_data_queue.clear()

    def exp_map(self, omega):
        angle = np.linalg.norm(omega)
        if angle < 1e-9:
            return np.eye(3)
        axis = omega / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.cos(angle) * np.eye(3) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * K

    def vectorToSkewSymmetric(self, vec):
        return np.array([[0, -vec[2], vec[1]],
                         [vec[2], 0, -vec[0]],
                         [-vec[1], vec[0], 0]], dtype=np.float32)

    def motionModelJacobian(self):
        dt = self.delta_t
        F = np.zeros((15, 15), dtype=np.float64)

        F[0:3, 9:12] = -np.eye(3)  
        F[3:6, 12:15] = -np.eye(3)  
        F[6:9, 3:6] = np.eye(3)  
        self.m_jacobian_matF = np.eye(15) + F * dt

    def motionModel(self):
        dt = self.delta_t
        R = self.state["R"]
        v = self.state["v"]
        p = self.state["p"]
        a_b = self.state["b_acc"]
        w_b = self.state["b_gyro"]
        acc = self.linear_acc - a_b
        omega = self.angular_vel - w_b

        R = R @ self.exp_map((omega * dt))
        v = v + (R @ acc + np.array([0, 0, 9.81])) * dt
        p = p + v * dt + 0.5 * (R @ acc + np.array([0, 0, 9.81])) * dt ** 2

        self.state["R"] = R
        self.state["v"] = v
        self.state["p"] = p

    def prediction(self):
        self.motionModel()
        self.motionModelJacobian()
        self.m_matP = self.m_jacobian_matF @ self.m_matP @ self.m_jacobian_matF.T + self.m_matQ

    def measurementModel(self):
        p = self.state["p"]
        for i in range(8):
            diff = p - self.uwb_position[:, i]
            self.m_vech[i] = np.linalg.norm(diff)

    def measurementModelJacobian(self):
        H = np.zeros((8, 15), dtype=np.float64)
        p = self.state["p"]
        for i in range(8):
            diff = p - self.uwb_position[:, i]
            dist = np.linalg.norm(diff)
            H[i, 6:9] = diff / dist  

        self.m_jacobian_matH = H

    def correction(self):
        self.measurementModel()
        self.measurementModelJacobian()

        residual = self.m_vecZ - self.m_vech
        residual_cov = self.m_jacobian_matH @ self.m_matP @ self.m_jacobian_matH.T + self.m_matR

        Kk = self.m_matP @ self.m_jacobian_matH.T @ np.linalg.inv(residual_cov)
        delta_xi = Kk @ residual

        delta_theta = delta_xi[0:3]
        delta_v = delta_xi[3:6]
        delta_p = delta_xi[6:9]
        delta_b_gyro = delta_xi[9:12]
        delta_b_acc = delta_xi[12:15]

        self.state["R"] = self.exp_map(delta_theta).T @ self.state["R"]
        self.state["v"] += delta_v
        self.state["p"] += delta_p
        self.state["b_gyro"] += delta_b_gyro
        self.state["b_acc"] += delta_b_acc

        self.m_matP = (np.eye(15) - Kk @ self.m_jacobian_matH) @ self.m_matP

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = self.state["p"][0]
        pose.pose.position.y = self.state["p"][1]
        pose.pose.position.z = self.state["p"][2]
        # quaternion = R.from_matrix(self.state["R"]).as_quat()
        quaternion = R.from_matrix(self.state["R"]).as_euler('xyz',degrees=True)
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        # pose.pose.orientation.w = quaternion[3]
        self.pub_ekf.publish(pose)

if __name__ == "__main__":
    rospy.init_node('se3_iekf')
    se3_iekf = SE3IEKF()
    rospy.spin()
