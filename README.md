# Height Mapper


Procedure:

1. Enumerate realsense devices and open D455
2. stream depth frames from d455
3. convert depth frames to point clouds
4. Feed point clouds to iterative closest point.
5. Build up global map from point clouds by registering subsequent point clouds into the map.
6. Estimate normals using k nearest neighbors.
7. Create gradient map from normal map
8. Convert gradient map to 2d image
9. Convert 2d image into obj file for use in simulation.

   

<img width="861" height="697" alt="image" src="https://github.com/user-attachments/assets/4229a0bd-f383-490e-b926-fc84925ac014" />
<img width="598" height="478" alt="image" src="https://github.com/user-attachments/assets/7e6a8d21-f156-4829-909b-aef0202fb5c5" />
<img width="492" height="478" alt="image" src="https://github.com/user-attachments/assets/d82a1d66-9f88-4b2d-8566-a16489ea91c6" />
<img width="1140" height="906" alt="image" src="https://github.com/user-attachments/assets/3d6d2ddb-3a72-47df-9a77-405f521830c0" />
