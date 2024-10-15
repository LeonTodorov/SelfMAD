import argparse
import torch
from model import Detector
from PIL import Image
import cv2
import numpy as np

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # MODEL
    model_state_path = args.model_path
    model = Detector(model=args.model_type)
    model_state = torch.load(model_state_path)
    model.load_state_dict(model_state['model'])
    model.train(mode=False)
    model.to(device)

    # IMG
    image_size = 384 if "hrnet" in args.model_type else 380
    img = np.array(Image.open(args.input_path))
    img = cv2.resize(img, (image_size, image_size))
    img=img.transpose((2,0,1))
    img = img.astype('float32')/255
    img = img[np.newaxis, ...]
    img = torch.from_numpy(img).to(device, non_blocking=True).float()
    with torch.no_grad():
        output = model(img).softmax(1)[:, 1].cpu().data.numpy()[0]
    print("Confidence:", output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-m', dest='model_type', type=str, required=True, help='Type of the model, e.g. hrnet_w18')
    parser.add_argument('-in', dest='input_path', type=str, required=True, help='Path to input image')
    parser.add_argument('-p', dest='model_path', type=str, required=True, help='Path to saved model')

    args = parser.parse_args()
    main(args)