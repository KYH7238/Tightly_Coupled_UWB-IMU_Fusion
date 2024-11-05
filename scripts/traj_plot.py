import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from nav_msgs.msg import Path
import tf2_ros

class Plotter:
    def __init__(self):
        self.esekf_x_uwb = []
        self.esekf_y_uwb = []
        self.esekf_z_uwb = []
        self.ukf_x_uwb = []
        self.ukf_y_uwb = []
        self.ukf_z_uwb = []
        self.ukf_esekf_x_uwb = []
        self.ukf_esekf_y_uwb = []
        self.ukf_esekf_z_uwb = []

        self.pf_x = []
        self.pf_y = []
        self.pf_z = []

        self.gt_x = []
        self.gt_y = []
        self.gt_z = []
        self.q_w = []
        self.q_x = []
        self.q_y = []
        self.q_z = []

        self.esekf_uwb_cnt = 0
        self.ukf_uwb_cnt = 0
        self.imu_cnt = 0
        self.cnt = 0

        self.path_1 = Path()
        self.path_1.header.frame_id = "map"
        
        self.path_2 = Path()
        self.path_2.header.frame_id = "map"
        
        self.path_3 = Path()
        self.path_3.header.frame_id = "map"
        
        self.path_gt = Path()
        self.path_gt.header.frame_id = "map"

        rospy.init_node("plotter")
        self.br = tf2_ros.TransformBroadcaster()
        self.pub_1 = rospy.Publisher('/uwb', Path, queue_size=10)
        self.pub_2 = rospy.Publisher('/ekf', Path, queue_size=10)
        self.pub_3 = rospy.Publisher('/esekf', Path, queue_size=10)
        self.pub_gt = rospy.Publisher('/gt', Path, queue_size=10)

        # rospy.Subscriber("/result_uwb", PoseStamped, self.esekf_uwb_callback)
        rospy.Subscriber("/result_ekf", PoseStamped, self.ukf_uwb_callback)
        # rospy.Subscriber("/result_uwb", PoseStamped, self.ukf_uwb_esekf_callback)
        rospy.Subscriber("/liekf", PoseStamped, self.ukf_uwb_esekf_callback)
        rospy.Subscriber("/estimated_path", Path, self.pf_callback)
        self.read_gt_file()

    def pf_callback(self, msg):
        self.pf_x.append(msg.poses[-1].pose.position.x)
        self.pf_y.append(msg.poses[-1].pose.position.y)
        self.pf_z.append(msg.poses[-1].pose.position.z+0.2)

    def esekf_uwb_callback(self, msg):
        # self.esekf_uwb_cnt += 1
        # if self.esekf_uwb_cnt >= 3:
        self.esekf_x_uwb.append(msg.pose.position.x)
        self.esekf_y_uwb.append(msg.pose.position.y)
        self.esekf_z_uwb.append(msg.pose.position.z+0.1)
            # self.esekf_uwb_cnt = 0

        self.publish_transform_and_path(msg, self.path_1, self.pub_1)

    def ukf_uwb_callback(self, msg):
        # self.ukf_uwb_cnt += 1
        # if self.ukf_uwb_cnt >= 3:
        self.ukf_x_uwb.append(msg.pose.position.x)
        self.ukf_y_uwb.append(msg.pose.position.y)
        self.ukf_z_uwb.append(msg.pose.position.z)
            # self.ukf_uwb_cnt = 0

        self.publish_transform_and_path(msg, self.path_2, self.pub_2)

    def ukf_uwb_esekf_callback(self, msg):
        # self.ukf_uwb_cnt += 1
        # if self.ukf_uwb_cnt >= 3:
        self.ukf_esekf_x_uwb.append(msg.pose.position.x)
        self.ukf_esekf_y_uwb.append(msg.pose.position.y)
        self.ukf_esekf_z_uwb.append(msg.pose.position.z)
            # self.ukf_uwb_cnt = 0

        self.publish_transform_and_path(msg, self.path_3, self.pub_3)

    def publish_transform_and_path(self, msg, path, pub):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = msg.header.frame_id

        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation

        self.br.sendTransform(t)

        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose

        path.poses.append(pose_stamped)
        path.header.stamp = rospy.Time.now()
        pub.publish(path)

    def read_gt_file(self):
        gt_file_path = "0612_hw3_gt.txt"  
        with open(gt_file_path, 'r') as file:
            for line in file:
                data = line.strip().split("\t")
                a, b, c, _, _, _, _, d, e, f, g, h, i, j, k, l = map(float, data)
                self.gt_x.append(a*0.001+4.55)
                self.gt_y.append(b*0.001+4)
                self.gt_z.append(c*0.001)
                rot = np.array([[d, e, f],
                                [g, h, i],
                                [j, k, l]])
                q = self.rotation_matrix_to_quaternion(rot)
                self.q_w.append(q[0])
                self.q_x.append(q[1])
                self.q_y.append(q[2])
                self.q_z.append(q[3])

                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = "map"
                pose_stamped.pose.position.x = a
                pose_stamped.pose.position.y = b
                pose_stamped.pose.position.z = c
                pose_stamped.pose.orientation.w = q[0]
                pose_stamped.pose.orientation.x = q[1]
                pose_stamped.pose.orientation.y = q[2]
                pose_stamped.pose.orientation.z = q[3]

                self.path_gt.poses.append(pose_stamped)

        self.path_gt.header.stamp = rospy.Time.now()
        self.pub_gt.publish(self.path_gt)

    def rotation_matrix_to_quaternion(self, R):
        q = np.empty((4,))
        t = np.trace(R)
        if t > 0.0:
            t = np.sqrt(t + 1.0)
            q[0] = 0.5 * t
            t = 0.5 / t
            q[1] = (R[2, 1] - R[1, 2]) * t
            q[2] = (R[0, 2] - R[2, 0]) * t
            q[3] = (R[1, 0] - R[0, 1]) * t
        else:
            i = 0
            if R[1, 1] > R[0, 0]:
                i = 1
            if R[2, 2] > R[i, i]:
                i = 2
            j = (i + 1) % 3
            k = (j + 1) % 3
            t = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0)
            q[i + 1] = 0.5 * t
            t = 0.5 / t
            q[0] = (R[k, j] - R[j, k]) * t
            q[j + 1] = (R[j, i] + R[i, j]) * t
            q[k + 1] = (R[k, i] + R[i, k]) * t
        return q

    def plot_data(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        while not rospy.is_shutdown():
            ax.clear()
            self.cnt += 1
            # ax.scatter(self.esekf_x_uwb, self.esekf_y_uwb, self.esekf_z_uwb, c='k', label='UWB position', s=1)
            ax.scatter(self.gt_x, self.gt_y, self.gt_z, c='k', label='Ground Truth', s=1)
            ax.scatter(self.ukf_x_uwb, self.ukf_y_uwb, self.ukf_z_uwb, c='r', label='EKF', s=1)
            ax.scatter(self.ukf_esekf_x_uwb, self.ukf_esekf_y_uwb, self.ukf_esekf_z_uwb, c='b', label='IEKF', s=1,linestyle = '-')
            # ax.scatter(self.pf_x, self.pf_y, self.pf_z, c='g', label='PF', s=1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('TIGHTLY Coupled Fusion')
            ax.legend()

            plt.pause(0.1)
            rospy.sleep(0.1)

if __name__ == "__main__":
    plotter = Plotter()
    # esekf.ESEKF()
    # nlink_se3.SE3EKF()
    plotter.plot_data()
    rospy.spin()
