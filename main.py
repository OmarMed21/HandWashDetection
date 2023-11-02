from utiles import *

def get_hkey(dict_):
    return max(dict_, key=dict_.get)

# cap = cv2.VideoCapture("test.mp4")
cap = cv2.VideoCapture(0)
plates = {}
detector = DetectWash()
tracker = Tracker(15, 20)
while cv2.waitKey(1) != 27 and cap.isOpened():
    _, frame = cap.read()
    frame = cv2.resize(frame, (640,480))
    hands = detector.get_hand_wash_coor(frame, 10)
    cars_data = tracker.update(hands.copy())
    print(cars_data)
    if len(cars_data)> 0:
        for tracker_id, (x1,y1,x2,y2), found, start_time in cars_data:
            if found:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
                cv2.putText(frame, f"{round(time.time()-start_time, 2)}", (x1,y1), 0, 2.8, (255,0,0), 2)
    cv2.imshow("frame", frame)
    

cap.release()
cv2.destroyAllWindows()
