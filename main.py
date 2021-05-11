import time
import cv2 as cv
from torchvision import transforms
from torchvision import models
import torch
from PIL import Image

print(dir(models))

alexnet = models.alexnet(pretrained=True)

print(alexnet)

transform = transforms.Compose([  # [1]
    transforms.Resize(256),  # [2]
    # transforms.CenterCrop(224),  # [3]
    transforms.ToTensor(),  # [4]
    transforms.Normalize(  # [5]
        mean=[0.485, 0.456, 0.406],  # [6]
        std=[0.229, 0.224, 0.225]  # [7]
    )])

webcam = cv.VideoCapture(0)

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

while (True):

    ret, webcam_frame = webcam.read()

    img_t = transform(Image.fromarray(webcam_frame))
    batch_t = torch.unsqueeze(img_t, 0)

    alexnet.eval()

    out = alexnet(batch_t)

    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    print(classes[index[0]], percentage[index[0]].item())
    _, indices = torch.sort(out, descending=True)
    [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

    # font which we will be using to display FPS
    font = cv.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)

    # puting the FPS count on the frame
    cv.putText(webcam_frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv.LINE_AA)

    cv.imshow('WEB Camera', webcam_frame)

    if (cv.waitKey(1) & 0xFF == ord('q')):
        break

webcam.release()
