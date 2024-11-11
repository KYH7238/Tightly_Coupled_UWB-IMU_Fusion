# Experimental Comparison of Tightly and Loosely Coupled Kalman Filters for 6D Pose Estimation Using UWB and IMU Sensors(2024 ICCAS)

we present a novel approach for robot pose estimation in SE(3) using a tightly coupled integration
of Ultra-Wideband (UWB) and Inertial Measurement Unit (IMU) data. Our method leverages Extended Kalman
Filter (EKF) and Error-State Kalman Filter (ESKF) to accurately estimate the pose, including position and orientation,
of a mobile robot in three-dimensional space. The tightly coupled approach ensures robust performance even in environments
where UWB data quality is poor or in NLOS(Non-Line-of-Sight) conditions by utilizing raw UWB data and
effectively fusing with IMU. The proposed method is validated through extensive simulations and real-world experiments,
demonstrating significant improvements in accuracy and robustness compared to the loosely coupled method. Our results
show that the tightly coupled approach, with its enhanced data fusion capability, provides superior z-axis estimation
performance. This makes it particularly advantageous in applications requiring accurate SE(3) pose estimation.

##  6D pose estimation using tightly coupled UWB/IMU fusion using filtering method(EKF/ESKF/UKF/LIEKF)
 - Extended Kalman filter
 - Error-state Kalman filter
 - Unscented  Kalman filter
 - Left-invariant Kalman filter

Experimental setup
---
- SBC: Intel NUC-i7
- UWB(50Hz): Nooploop LinkTrack P-b (1tag, 8anchors)
- IMU(20Hz): UM7
- Ground Truth data: Qualisys Arqus motion capture system(10ea)


![setup](https://github.com/user-attachments/assets/8e8b76ed-4682-434a-b2a4-71f2e6146352)
**This system included one tag and eight anchors configured in a rectangular cuboid formation.**   
   
*If you want to test in your environment, you need to change the config file(config/params.yaml).*

    cd catkin_workspace/src    
    git clone --recursive https://github.com/nooploop-dev/nlink_parser.git 
    git clone https://github.com/KYH04444/Tightly_Coupled_Fusion.git
    cd ..   
    catkin_make
---

    roslaunch tightly_coupled all.launch
    


Experimental results
---
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/1370cd3e-14b3-49e6-a939-375e31f7ed91" alt="tightly_efk_esekf" width="500" height="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/57946b79-5339-4f88-a661-2233f41ccb7a" alt="tightlycoupled_test2" width="500" height="400"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c1918811-18ef-459e-8555-a2d3a52b86ed" alt="tightly_position" width="500" height="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/5c8969e6-d32b-4976-a2e7-ff16d26474d8" alt="tightly_orien" width="500" height="400"/></td>
  </tr>
</table>



