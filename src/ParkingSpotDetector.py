import cv2
import numpy as np
import time

video_path = "video/CarPark.mp4"
cap = cv2.VideoCapture(video_path)
parking_slots = [(48,110), (44,153), (43,201), (40,249), (40,299), (41,344), (40,392), (45,439), (42,483), (45,532), (46,582), (47,628), (171,626), (168,578), (167,536), (171,485), (165,436), (165,388), (168,342), (168,299), (170,247), (166,202), (163,157), (164,104), (389,104), (390,149), (391,199), (391,245), (391,292), (392,340), (391,389), (392,434), (396,530), (395,577), (394,626), (513,626), (516,579), (512,533), (508,434), (512,388), (511,335), (507,293), (509,245), (512,196), (511,150), (512,100), (732,99), (735,145), (742,190), (739,240), (737,293), (738,339), (740,381), (739,433), (743,479), (744,524), (744,573), (746,626), (909,624), (907,576), (908,531), (904,483), (903,438), (907,386), (907,343), (904,292), (906,243), (907,196), (901,155)]
rect_width, rect_height = 92, 30
# rect_width, rect_height = 0, 0
global i, j
color = (0,0,255)
thick = 1
threshold = 30
last_call_time = time.time()
prevFreeslots = 0
i = 0
j = 0

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'({x},{y})')
        parking_slots.append((x, y))



def convert_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = frame.copy()
    contour_image[:] = 0 
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), thickness=2)
    return contour_image

def mark_slots(frame, grayscale_frame, key):
    global last_call_time
    global prevFreeslots
    global i, j
    current_time = time.time()
    elapsed_time = current_time - last_call_time

    freeslots=0
    if rect_height:
        for x, y in parking_slots:
            x1=x
            x2=x+rect_width
            y1=y
            y2=y+rect_height
            start_point, stop_point = (x1,y1), (x2, y2)

            crop = grayscale_frame[y1:y2, x1:x2]
            crop_frame = frame[y1:y2, x1:x2]
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            count=cv2.countNonZero(gray_crop)

            color, thick = [(0,255,0), 2] if count < threshold else [(0,0,255), 2]

            if count < threshold:
                freeslots = freeslots+1
                cv2.imwrite(f"Data/Free/{i + 1}.jpg", crop_frame)
                i += 1
            else:
                cv2.imwrite(f"Data/Parked/{j + 1}.jpg", crop_frame)
                j += 1

            
            cv2.rectangle(frame, start_point, stop_point, color, thick)

    if key:
        current_time = time.time()
        if current_time - last_call_time >= 0.1:
            cv2.putText(frame, "Free Slots:" + str(freeslots), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 255), 2)
            last_call_time = current_time
            prevFreeslots = freeslots
        else:
            cv2.putText(frame, "Free Slots:" + str(prevFreeslots), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 255), 2)

    return frame

cv2.namedWindow("Parking Spot Detector")
cv2.setMouseCallback("Parking Spot Detector", click_event)

ret, frame = cap.read()
key = False

while True:
        if key:
            ret, frame = cap.read()

            if not ret:break

            grayscale_frame = convert_grayscale(frame)
            out_image = mark_slots(frame, grayscale_frame, key)
            cv2.imshow("Parking Spot Detector", out_image)     
            # cv2.imshow("Parking Spot Detector", grayscale_frame)
        
            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                break
            if k & 0xFF == ord('a'):
                key = False
        
        else:
            if(len(parking_slots) == 2 and not rect_height):
                rect_width, rect_height = parking_slots[1][0] - parking_slots[0][0], parking_slots[1][1] - parking_slots[0][1]
                print(parking_slots)
                print(rect_width, rect_height)
                del parking_slots[1]
                
            grayscale_frame = convert_grayscale(frame)
            out_image = mark_slots(frame, grayscale_frame, key)  
            cv2.imshow("Parking Spot Detector", out_image)

            k = cv2.waitKey(1)
            if k & 0xFF == ord('a'):
                key = True
            if k & 0xFF == ord('z'):
                del parking_slots[-1]
            if k & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
