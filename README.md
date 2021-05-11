# OpenCV-CUDA

##### System-Configuration

Device 0: "GeForce GTX 1650"
* CUDA Driver Version / Runtime Version:         11.0 / 11.0
* CUDA Capability Major/Minor version number:    7.5
* Total amount of global memory:                 4096 MBytes (4294967296 bytes)
* GPU Clock Speed:                               1.56 GHz
* Warp size:                                     32
* Maximum number of threads per block:           1024
* Maximum sizes of each dimension of a block:    1024 x 1024 x 64
* Maximum sizes of each dimension of a grid:     2147483647 x 65535 x 65535
* Maximum memory pitch:                          2147483647 bytes
* Texture alignment:                             512 bytes
* Concurrent copy and execution:                 Yes with 6 copy engine(s)
* Run time limit on kernels:                     Yes
* Compute Mode:                                  Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)


##### Pre-Requisites for Usage follow  
* https://jamesbowley.co.uk/accelerate-opencv-4-4-0-build-with-cuda-and-python-bindings/

References

* https://forums.developer.nvidia.com/t/translating-cpu-based-opencv-code-to-gpu-based-opencv-code/51369
* https://github.com/Kjue/python-opencv-gpu-video
* https://docs.opencv.org/2.4/modules/gpu/doc/per_element_operations.html
* https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/

------------------------------------------------------------------------------------------------------------------------ 
 
### Test 1
Running addition on two captured image from webcams

Using 2 real time image captures from the webcams of Resolution 640x480

##### Observation
The CPU execution is faster than the GPU by a factor of 20%

Reasons
* The add is a small operation, and any performance boost you get from doing it on the GPU is vastly outweighed by memory transfer times between host (CPU) and device (GPU). Minimizing the latency of this memory transfer is a primary challenge of any GPU computing.

References for studying
* https://stackoverflow.com/questions/12074281/why-opencv-gpu-code-is-slower-than-cpu
* https://www.geeks3d.com/20100606/gpu-computing-nvidia-cuda-compute-capability-comparative-table/#:~:text=The%20Compute%20Capability%20describes%20the,t%20start%20on%20your%20system.
* https://docs.opencv.org/2.4/modules/gpu/doc/introduction.html

##### Rev 1.0.1 (Test)

- Trying Alexnet with Webcam feed. 
- Displaying the classified label for the same. 