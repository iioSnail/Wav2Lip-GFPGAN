import os
import argparse

import cv2
from tqdm import tqdm
from os import path
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--audio', type=str, required=True)
parser.add_argument('--outfile', type=str, required=True)
parser.add_argument('--face', type=str, required=True)

args = parser.parse_args()

wav2lipFolderName = 'Wav2Lip-master'
gfpganFolderName = 'GFPGAN-master'
basePath = "/root/Wav2Lip-GFPGAN"

wav2lipPath = basePath + '/' + wav2lipFolderName
gfpganPath = basePath + '/' + gfpganFolderName

outputPath = basePath + '/temp'
inputAudioPath = args.audio
inputVideoPath = args.face
lipSyncedOutputPath = basePath + '/temp/wav2lip_output.mp4'


shutil.rmtree(outputPath)

if not os.path.exists(outputPath):
    os.makedirs(outputPath)

cmd = (f"python ./Wav2Lip-master/inference.py "
       f"--checkpoint_path ./Wav2Lip-master/checkpoints/wav2lip.pth "
       f"--face {inputVideoPath} "
       f"--audio {inputAudioPath} "
       f"--outfile {lipSyncedOutputPath}")
os.system(cmd)

inputVideoPath = lipSyncedOutputPath
unProcessedFramesFolderPath = outputPath + '/frames'

if not os.path.exists(unProcessedFramesFolderPath):
    os.makedirs(unProcessedFramesFolderPath)

vidcap = cv2.VideoCapture(inputVideoPath)
numberOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vidcap.get(cv2.CAP_PROP_FPS)
print("FPS: ", fps, "Frames: ", numberOfFrames)

for frameNumber in tqdm(range(numberOfFrames)):
    _, image = vidcap.read()
    cv2.imwrite(path.join(unProcessedFramesFolderPath, str(frameNumber).zfill(4) + '.jpg'), image)
    

cmd = (f"python ./GFPGAN-master/inference_gfpgan.py -i {unProcessedFramesFolderPath} -o {outputPath} "
       f"-v 1.3 -s 2 --only_center_face --bg_upsampler None")

os.system(cmd)

restoredFramesPath = outputPath + '/restored_imgs/'
processedVideoOutputPath = outputPath

dir_list = os.listdir(restoredFramesPath)
dir_list.sort()

batch = 0
batchSize = 300

for i in tqdm(range(0, len(dir_list), batchSize)):
    img_array = []
    start, end = i, i + batchSize
    print("processing ", start, end)
    for filename in tqdm(dir_list[start:end]):
        filename = restoredFramesPath + filename;
        img = cv2.imread(filename)
        if img is None:
            continue

        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(processedVideoOutputPath + '/batch_' + str(batch).zfill(4) + '.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    batch = batch + 1

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

concatTextFilePath = outputPath + "/concat.txt"
concatTextFile = open(concatTextFilePath, "w")

for ips in range(batch):
    concatTextFile.write("file batch_" + str(ips).zfill(4) + ".avi\n")

concatTextFile.close()

concatedVideoOutputPath = outputPath + "/concated_output.avi"

cmd = f"ffmpeg -y -f concat -i {concatTextFilePath} -c copy {concatedVideoOutputPath}"

os.system(cmd)

finalProcessedOuputVideo = args.outfile

cmd = (f"ffmpeg -y -i {concatedVideoOutputPath} "
       f"-i {inputAudioPath} -map 0 -map 1:a -c:v copy "
       f"{finalProcessedOuputVideo}")

os.system(cmd)

print("Finished!!1")
