import os
import lmdb
import cv2

import numpy as np

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath='Data/ImageNet/train', outputPath='Data/ImageNet_lmbd/train', checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=23690000000)
    cache = {}
    cnt = 1

    for l_idx, label in enumerate(os.listdir(inputPath)):
        imgs_path = os.path.join(inputPath, label)
        if not os.path.exists(imgs_path):
            print('%s does not exist' % imgs_path)
            continue
        for i_idx, img in enumerate(os.listdir(imgs_path)):
            img = os.path.join(imgs_path, img)
            with open(img, 'rb') as f:
                imageBin = f.read()
            if checkValid:
                try:
                    if not checkImageIsValid(imageBin):
                        print('%s is not a valid image' % img)
                        continue
                except:
                    print('error occured', l_idx, i_idx)
                    continue

            imageKey = 'image-%09d'.encode() % cnt
            labelKey = 'label-%09d'.encode() % cnt
            cache[imageKey] = imageBin
            cache[labelKey] = label.encode()

            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d - %d' % (l_idx, i_idx))
            cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

if __name__ == '__main__':
    createDataset();