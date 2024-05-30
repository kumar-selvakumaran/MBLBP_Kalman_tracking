# MBLBP based tracking with Kalman filtering - Prototype

- Writing a python implementation as a prequel to C++.
- Figuring out all engineering challenges before facing C++ challenges as a somewhat-beginner.

environment:
- Docker : <code>pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime</code>

Docker run : <code> docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --network=host --volume=/mnt/d/projects/MBLBP_Kalman_tracking/python_implementation/app:/app --workdir=/app pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime /bin/bash
</code>

<hr>
## Algorithm overview 

components of a general searching algorithm : 
- appearance model : to identify the features that distinguish the target object from other objects
- Motion Model : to model the motion of the target object. This essentially allows us to predict the next possible location of the object in a video.
- Searching Strategy : An efficient strategy searching for the target in relevant regions of the image  when the object is lost.