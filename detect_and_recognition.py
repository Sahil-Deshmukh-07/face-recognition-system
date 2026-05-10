import cv2
import dlib
import face_recognition as fr

detector = dlib.get_frontal_face_detector()

known_face_1 = fr.load_image_file(r"C:\Codes\Face Recognition\Sahil.jpg")
known_face_2 = fr.load_image_file(r"C:\Codes\Face Recognition\Kshitij.jpg")
known_encoding_1 = fr.face_encodings(known_face_1)[0]
known_encoding_2 = fr.face_encodings(known_face_2)[0]

knwon_face_encoding = [
    known_encoding_1,
    known_encoding_2
]
known_face_names = [
    "Sahil",
    "Kshitij"
]


video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    RGB_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    faces_location = fr.face_locations(RGB_frame)
    faces_encoding = fr.face_encodings(RGB_frame,faces_location)
    frame = cv2.flip(frame,1)

    for (y,x2,y2,x),face_encoding in zip(faces_location,faces_encoding):
        frame_width = frame.shape[1]
        matches = fr.compare_faces(knwon_face_encoding,face_encoding)
        name = "Unknown Person"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        left = frame_width - x2
        right = frame_width - x
        cv2.rectangle(frame,(right,y),(left,y2),(0,255,0),2)
        cv2.putText(frame,name,(right,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()