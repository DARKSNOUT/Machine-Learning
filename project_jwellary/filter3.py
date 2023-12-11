import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def find_middle_point(a, b):
    return (((a[0] + b[0]) / 2), ((a[1] + b[1]) / 2), ((a[2] + b[2]) / 2))

def apply_snap_filter(face_img, filter_img, x, y, w, h, scale_factor):
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    filter_img = cv2.resize(filter_img, (new_w, new_h))
    face_img2=face_img.copy()
    alpha_mask = filter_img[:, :, 3] / 255.0
    inv_alpha_mask = 1.0 - alpha_mask
    x_offset = int((w - new_w) / 2)
    y_offset = int((h - new_h) / 2)
    for c in range(0, 3):
        face_img[y+y_offset:y+y_offset+new_h, x+x_offset:x+x_offset+new_w, c] = (alpha_mask * filter_img[:, :, c] + inv_alpha_mask * face_img[y+y_offset:y+y_offset+new_h, x+x_offset:x+x_offset+new_w, c])

    return face_img

cap = cv2.VideoCapture(0)
image2 = cv2.imread(r'D:\ML\Mediapipe\1.png',-1)
#image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
#image2=cv2.resize(image2,(288,352))
#print(image2.shape)
if image2 is None:
    raise FileNotFoundError(f"Error: Image not found at path ")

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        #print("shape_frame", frame.shape)

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Check if pose_landmarks is present
        if results.pose_landmarks:
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # try:
            landmarks = results.pose_landmarks.landmark
        
            a = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z])
            c = np.array([landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y, landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].z])
            b = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z])
            d = np.array([landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y, landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].z])

            # Calculate middle points
            mac = find_middle_point(a*frame.shape[0], c*frame.shape[1])
            mbd = find_middle_point(b*frame.shape[0], d*frame.shape[1])
            a1=[a[0]*frame.shape[0],a[1]*frame.shape[1],a[2]*frame.shape[2]]
            c1=[c[0]*frame.shape[0],c[1]*frame.shape[1],c[2]*frame.shape[2]]
            b1=[b[0]*frame.shape[0],b[1]*frame.shape[1],b[2]*frame.shape[2]]
            d1=[d[0]*frame.shape[0],d[1]*frame.shape[1],d[2]*frame.shape[2]]
            m=mac[0]-mbd[0]
            ab=a1[0]-b1[0]
            print(mbd[1],d1[1])
            # print("mac", mac)
            # print("mbd", mbd)

            # Convert coordinates to integers
            mac_int = tuple(map(int, mac))
            mbd_int = tuple(map(int, mbd))
            """
            image=cv2.circle(image, (mac_int[0],mac_int[1]),2,(0,255,255),-1)
            image=cv2.circle(image, (mbd_int[0],mbd_int[1]),2,(0,255,255),-1)"""

            """for landmark in landmarks:
                    x, y, z = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]), int(landmark.z * frame.shape[1])
                    cv2.circle(image, (x, y), 2, (0, 0, 255), -1)"""

            #print(mac[0],mac[1],m,100)
            img=apply_snap_filter(image, image2,int(((b1[0]+a1[0])/2)-2),int(mbd[1]-10),int(ab/2),int(ab/2),1.5)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # except Exception as e:
            #     img = frame
            #     print(f"Error: {e}")
        else:
            img=frame
        # Display the image
        # img=apply_snap_filter(image, image2,100, 100, 100, 100,1.2)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Overlay", img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        # break

    cap.release()
    cv2.destroyAllWindows()
