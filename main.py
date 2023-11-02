from utiles import *

cap = cv2.VideoCapture("test.mp4")
detector = DetectWash()
tracker = Tracker(15, 50)

all_poped_hands = {}

while cv2.waitKey(1) != 27 and cap.isOpened():
    _, frame = cap.read()
    frame = cv2.resize(frame, (640,480))
    hands = detector.get_hand_wash_coor(frame, 10)
    hands_data = tracker.update(hands.copy())
    poped_objects = tracker.poped_objects
    for poped_object in poped_objects:
        time_taken_by_one_hand_wash = time.time()-poped_object["start-time"]
        fomated_date_of_start_time = unix_time2data(poped_object["start-time"])
        if time_taken_by_one_hand_wash > 1: # if time taken is more than 1 sec
            all_poped_hands[fomated_date_of_start_time] = round(time_taken_by_one_hand_wash, 4)
    print(all_poped_hands)
    if len(hands_data) > 0:
        for tracker_id, (x1,y1,x2,y2), found, start_time in hands_data:
            if found:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
                cv2.putText(frame, f"{round(time.time()-start_time, 2)}", (x1,y1), 0, 2.8, (255,0,0), 2)
    cv2.imshow("frame", frame)

cap.release()
cv2.destroyAllWindows()