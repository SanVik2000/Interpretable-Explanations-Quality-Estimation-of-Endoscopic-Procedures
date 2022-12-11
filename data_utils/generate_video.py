import cv2
import numpy as np
import os
from os.path import isfile, join
import argparse
from datetime import datetime
from PIL import Image
import sys

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files= [file for file in os.listdir(pathIn) if file.endswith('.png')]
    
    files.sort(key = lambda x: int(x.split('_')[1].split('.')[0]))
    
    for i in range(len(files)):
        filename=pathIn + '/' + files[i]

        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])

    out.release()

def plot_img(pathIn):
    files= [file for file in os.listdir(pathIn) if file.endswith('.png')]
    files.sort(key = lambda x: int(x.split('_')[1].split('.')[0]))
    images = [Image.open(os.path.join(pathIn,x)) for x in files][15:50]
    
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in images])[0][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in images ) )
    
    imgs_comb = Image.fromarray( imgs_comb).convert('RGB')
    imgs_comb.save( os.path.join(pathIn,'all.jpg') )  

def vertical_stack(path1, path2, path3, path4, out_dir):
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    img3 = Image.open(path3)
    img4 = Image.open(path4)
    images= [img1, img2, img3, img4]
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in images])[0][1]
    imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in [img1, img2, img3, img4] ) )
    imgs_comb = Image.fromarray( imgs_comb).convert('RGB')
    imgs_comb.save( os.path.join(out_dir,'vis.jpg') )  

def add_args(parser):
    parser.add_argument("-o", "--out_dir", type=str, default=f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}", help="path to output directory [default: year-month-date_hour-minute]",)
    return parser

def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    pathIn= args.out_dir
    pathOut = os.path.join(args.out_dir, 'video.mp4')
    fps = 10.0
    convert_frames_to_video(pathIn, pathOut, fps)
    plot_img(pathIn)

def generate(dir):

    pathIn= dir
    plot_img(pathIn)

if __name__=="__main__":
    main()