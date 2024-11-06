# Tightly Coupled UWB/IMU Fusion

 6D pose estimation using tightly coupled UWB/IMU fusion using filtering method(EKF/ESKF/UKF/LIEKF)
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

    roslaunch tightly_coupled ekf.launch
    


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

## Experimental Comparison of Tightly and Loosely Coupled Kalman Filters for 6D Pose Estimation Using UWB and IMU Sensors
 Paper Will be uploaded..
