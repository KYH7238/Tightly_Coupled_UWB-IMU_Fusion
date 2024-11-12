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
    <td><img src="https://github.com/user-attachments/assets/8ab2a293-5528-4227-8b20-aa25ec5e443a" alt="tightlycoupled_test2" width="500" height="400"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/8f14ac81-7234-4588-b597-745330b57fa3" alt="tightly_position" width="500" height="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/27a72a4b-c102-4329-934d-ee003716b5f7" alt="tightly_orien" width="500" height="400"/></td>
  </tr>
</table>

---
### Position RMSE (meter)

| Filter | X RMSE | Y RMSE | Z RMSE |
|--------|--------|--------|--------|
| **EKF**   | 0.1573 | 0.1211 | 0.1479 |
| **ESKF**  | 0.1573 | 0.1212 | 0.1480 |
| **LIEKF** | 0.1567 | 0.1204 | 0.1513 |
| **UKF**   | 0.2446 | 0.2083 | 0.1285 |
### Orientation RMSE (degree)

| Filter | Roll RMSE | Pitch RMSE | Yaw RMSE |
|--------|-----------|------------|----------|
| **EKF**   | 9.2247 | 8.8638 | 33.5447 |
| **ESKF**  | 9.2231 | 8.8654 | 15.8335 |
| **LIEKF** | 8.5259 | 9.2223 | 9.7865  |
| **UKF**   | 9.5407 | 9.3895 | 17.0105 | 


