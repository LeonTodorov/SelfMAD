import os
import cv2
import numpy as np
import dlib
from imutils import face_utils
from tqdm import tqdm
import argparse
from pathlib import Path

def crop_face(img, landmark):
    h, w = len(img),len(img[0])
    x0,y0 = landmark[:, 0].min(), landmark[:, 1].min()
    x1,y1 = landmark[:, 0].max(), landmark[:, 1].max()
    w_margin = w * 0.0625
    h_margin = h * 0.0625
    x0_new = max(0, int(x0 - w_margin))
    x1_new = min(w, int(x1 + w_margin))
    y0_new = max(0, int(y0 - h_margin))
    y1_new = min(h, int(y1 + h_margin))
    return img[y0_new:y1_new, x0_new : x1_new]

def save_crop(org_path, save_path, face_detector, face_predictor, forget=False):
    frame = cv2.imread(org_path, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame is None:
        raise ValueError(f"Image at path {org_path} could not be read.")
    faces = face_detector(frame, 1)
    if len(faces) == 0:
        print('No faces in {}'.format(org_path))
        return 
    landmarks = []
    size_list = []
    for face_idx in range(len(faces)):
        landmark = face_predictor(frame, faces[face_idx])
        landmark = face_utils.shape_to_np(landmark)
        x0,y0=landmark[:, 0].min(), landmark[:, 1].min()
        x1,y1=landmark[:, 0].max(), landmark[:, 1].max()
        face_s = (x1 - x0) * (y1 - y0)
        size_list.append(face_s)
        landmarks.append(landmark)
    landmarks = np.concatenate(landmarks).reshape((len(size_list),) + landmark.shape)
    landmarks = landmarks[np.argsort(np.array(size_list))[::-1]]
    frame_cropped = crop_face(frame, landmarks)
    print(f"{org_path}({frame.shape}) -> {save_path}({frame_cropped.shape})")
    if not forget:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2BGR))
    
    if len(landmarks) == 0:
        print('No landmarks in {}'.format(org_path))
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, help='Path to image daqtaset', required=True)
    parser.add_argument('-f', '--forget', action='store_true', default=False, help='If specified, do not save results', required=False)
    args = parser.parse_args()
    
    
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = 'utils/preprocessing/shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)
    
    input_path = Path(args.input_path).expanduser().resolve()
    output_path = str(Path(input_path).parent / (Path(input_path).name + "_cropped"))
    
    for root, _, files in os.walk(input_path):
        for file in tqdm(files, desc=f"Processing images in {root}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_path)
                save_dir = os.path.join(output_path, rel_path)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, file)
                save_crop(img_path, save_path, face_detector, face_predictor, forget=args.forget)
