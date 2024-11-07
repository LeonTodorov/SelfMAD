from faceparser import FaceParser
import os
import numpy as np
import torch
import argparse
from PIL import Image

def create_face_labels(img_pth, save_path, face_parser):
    img = np.array(Image.open(img_pth), dtype=np.uint8)
    face_labels = face_parser.parse(Image.fromarray(img))
    # print(img, img_pth, save_path, face_labels.shape)
    np.save(save_path, face_labels)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp = FaceParser(device=device)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, help='Path to image dataset', required=True)
    parser.add_argument('-o', '--output_name', type=str, help='Name appended to output', required=True)
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
                create_face_labels(img_pth, save_path, fp)
