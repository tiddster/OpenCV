import face_recognition
import cv2
from numba import jit

@jit
def faceLocAndEncode(img):
    _ = face_recognition.face_locations(img)
    if len(_) == 0:
        return None
    loc = _[0]
    encode = face_recognition.face_encodings(img)[0]
    cv2.rectangle(img, (loc[3], loc[0]), (loc[1], loc[2]), (0, 0, 255), 2)
    return encode

@ jit
def compare_faces(encodeList, aimEncode):
    return face_recognition.compare_faces(encodeList, aimEncode)

@jit
def face_distance(encodeList, aimEncode):
    return face_recognition.face_distance(encodeList, aimEncode)
