import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import numpy as np
from scipy.spatial.transform import Rotation as R

def read_ground_truth(file_path):
    positions = []
    orientations = []
    with open(file_path, 'r') as file:
        for line in file:
            data_line = line.strip().split("\t")
            if len(data_line) < 16:
                continue  
            a, b, c, _, _, _, _, d, e, f, g, h, i, j, k, l = map(float, data_line)
            x = a * 0.001 + 4.55
            y = b * 0.001 + 4.00
            z = c * 0.001
            rot_matrix = np.array([
                [d, e, f],
                [g, h, i],
                [j, k, l]
            ])
            rotation = R.from_matrix(rot_matrix)
            quat = rotation.as_quat()  # [x, y, z, w] 순서
            positions.append([x, y, z])
            orientations.append(quat)
    return positions, orientations

def publish_ground_truth(positions, orientations, total_duration):
    rospy.init_node('ground_truth_publisher', anonymous=True)
    pub = rospy.Publisher('/ground_truth', PoseStamped, queue_size=10)

    total_points = len(positions)
    if total_points == 0:
        rospy.logerr("No ground truth data to publish.")
        return

    time_interval = total_duration / total_points

    rate = rospy.Rate(1.0 / time_interval)
    start_time = rospy.Time.now()

    for i in range(total_points):
        if rospy.is_shutdown():
            break
        pose = PoseStamped()
        pose.header = Header()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"
        pose.pose.position.x = positions[i][0]
        pose.pose.position.y = positions[i][1]
        pose.pose.position.z = positions[i][2]
        pose.pose.orientation.x = orientations[i][0]
        pose.pose.orientation.y = orientations[i][1]
        pose.pose.orientation.z = orientations[i][2]
        pose.pose.orientation.w = orientations[i][3]

        pub.publish(pose)
        rate.sleep()

    elapsed_time = (rospy.Time.now() - start_time).to_sec()
    if elapsed_time < total_duration:
        rospy.sleep(total_duration - elapsed_time)

if __name__ == "__main__":
    file_path = '../config/0612_hw3_gt.txt'  
    total_duration = 99.719699  

    positions, orientations = read_ground_truth(file_path)
    publish_ground_truth(positions, orientations, total_duration)
