import os
import cv2
import numpy as np
import argparse
import dlib
from imutils import face_utils


def save_landmarks(org_path, save_path, face_detector, face_predictor):
    frame = cv2.imread(org_path, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # print(frame.shape)
    if frame is None:
        raise ValueError(f"Image at path {org_path} could not be read.")
    faces = face_detector(frame, 1)
    if len(faces) == 0:
        print('No faces in {}'.format(org_path))
        return 
    face_s_max=-1
    landmarks=[]
    size_list=[]
    for face_idx in range(len(faces)):
        landmark = face_predictor(frame, faces[face_idx])
        landmark = face_utils.shape_to_np(landmark)
        x0,y0=landmark[:,0].min(),landmark[:,1].min()
        x1,y1=landmark[:,0].max(),landmark[:,1].max()
        face_s=(x1-x0)*(y1-y0)
        size_list.append(face_s)
        landmarks.append(landmark)
    landmarks=np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
    landmarks=landmarks[np.argsort(np.array(size_list))[::-1]]

    # print(f"Org path: {org_path}")
    # print(f"Save path: {save_path}")
    # print(f"Landmarks shape: {landmarks.shape}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, landmarks)

if __name__ == "__main__":
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = 'shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, help='Path to image dataset', required=True)
    parser.add_argument('-o', '--output_name', type=str, help='output_name', required=True, default='_landmarks')
    args = parser.parse_args()
            
    input_path = args.input_path
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                img_pth = os.path.join(root, file)
                datadir = os.path.basename(os.path.normpath(input_path))
                new_dir = root.replace(datadir, datadir + "_" + args.output_name)
                save_path = os.path.join(new_dir, file)
                save_path = save_path.replace('.png', '.npy')
                save_path = save_path.replace('.jpg', '.npy')
                save_landmarks(img_pth, save_path, face_detector, face_predictor)