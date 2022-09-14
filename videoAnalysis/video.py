import cv2
import numpy as np

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print('Unable to open camera.')

frame_width = int(camera.get(3))
frame_height = int(camera.get(4))

print(frame_width,frame_height)

out = cv2.VideoWriter('COMP484/Facial-Landmark-Detection/CV/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

# def crop(image):
#     out = np.empty([224,224])
#     e = 0
#     for i in image[128:352]:
#         c = list(i)
#         c = c[208:432]
#         c = np.asarray(c)
#         out[e] = c
#         e += 1
#     return out
cutout = [*range(0,208)]
cutout.extend([*range(432,640)])
print(cutout)
def crop(image):
    out = image.copy()
    out = np.delete(out, cutout, axis = 1)
    out = out[128:352]/255.0
    return out


while(True):
    ret, frame = camera.read()
    if ret == True:
    # Write the frame into the file 'output.avi'
        out.write(frame)
        # Display the resulting frame  
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
        gray = crop(gray)

        cv2.imshow('frame',gray)
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Break the loop
    else:
        break 

camera.release()
out.release()
cv2.destroyAllWindows()