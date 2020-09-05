import cv2 as cv

webcam = cv.VideoCapture(0)
dedcam = cv.VideoCapture(1)

while (True):
    ret, webcam_frame = webcam.read()
    ret, dedcam_frame = dedcam.read()

    webcam_gpu_frame = cv.cuda_GpuMat(webcam_frame)
    dedcam_gpu_frame = cv.cuda_GpuMat(dedcam_frame)

    # print(webcam_gpu_frame.size())
    # print(dedcam_gpu_frame.size())

    added_gpu_frame = cv.cuda_GpuMat(webcam_frame)

    cv.cuda.add(webcam_gpu_frame, dedcam_gpu_frame, added_gpu_frame)



    cv.imshow('WEB Camera', webcam_frame)
    cv.imshow('USB Camera', dedcam_frame)
    cv.imshow('Added Image', cv.(added_gpu_frame))

    if cv.waitKey(1) == 'q':
        break

webcam.release()
dedcam.release()
cv.destroyAllWindows()