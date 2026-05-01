This is a sub branch from New-LiDAR-Obstacle-Detector Branch upon the lane detection accuracy reduced significantly.

Current Issues in this commit
1. LiDAR measured distance is not stable, DRIVE and SLOW changes frequently causing unstable decision making. I think giving a threshold value 
2. LiDAR detection only works on automatic mode. not in manual mode.
3. Traffic Light model is not accurate, even without the traffic light, it is detecting
