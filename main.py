import cv2 as cv
import time

webcam = cv.VideoCapture(0)
dedcam = cv.VideoCapture(1)

while(True):

    ret, webcam_frame = webcam.read()
    ret, dedcam_frame = dedcam.read()

    cv.imshow('WEB Camera', webcam_frame)
    cv.imshow('USB Camera', dedcam_frame)

    print("CPU Section")
    added_frame = webcam_frame.copy()
    added_frame = webcam_frame.copy()
    start = cv.getTickCount()
    cv.add(webcam_frame, dedcam_frame, added_frame)
    end = cv.getTickCount()
    print("Time Elapsed ",(end-start)/cv.getTickFrequency())
    cv.imshow('CPU Added Image', added_frame)

    print("GPU Section")
    webcam_gpu_frame = cv.cuda_GpuMat(webcam_frame)
    dedcam_gpu_frame = cv.cuda_GpuMat(dedcam_frame)
    added_gpu_frame = cv.cuda_GpuMat(webcam_frame)

    start = cv.getTickCount()
    cv.cuda.add(webcam_gpu_frame, dedcam_gpu_frame, added_gpu_frame)
    end = cv.getTickCount()

    print("Time Elapsed ",(end-start)/cv.getTickFrequency())

    cv.imshow('GPU Added Image', added_gpu_frame.download())

    if(cv.waitKey(1) & 0xFF == ord('q')):
        break

webcam.release()
dedcam.release()
cv.destroyAllWindows()