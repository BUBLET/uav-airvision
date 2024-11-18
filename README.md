# uav-airvision

**uav-airvision** is a monocular visual odometry system for UAVs, currently not ready.

## Project Overview
The system aims to estimate the uav`s camera pose and trajectory using:
- ORB-based feature extraction and matching.
- Essential and Homography matrices for initial pose estimation.
- 3D map point triangulation and tracking.
- Continuous pose updates via PnP with RANSAC.

## Current Status
The project is in progress:
- [x] Feature extraction and matching.
- [x] Initial pose estimation using Essential/Homography matrices.
- [x] Local Map.
- [ ] Basic adjustment.
- [ ] Real-time tracking and optimization.
- [ ] Integrate IMU sensors
- [ ] Testing on uav

## 🔮 Future Goals
- Improve real-time performance.
- Test on real UAV footage.
