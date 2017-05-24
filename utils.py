import numpy as np
import h5py
import scipy.misc
from PIL import Image
from model import * 
from os import listdir
from os.path import isfile, join
import time
import lasagne

path = '/NOBACKUP/dash_cam_dataset/'


def load_dataset_dummy():
    return np.ones((1000, 3, 64, 64))

def load_dataset(ds_split=(0.95,0.05,0.), shuffle=False, sample=False, resize=False, normalize=False, extra=False, easy=True):
    print("Loading dataset")
    home = '/home/ml/lpagec/'
    name = 'data.bin' if normalize else 'data_og.bin'
    try :
        f = file(home + name,"rb")
        trainz = np.load(f)
        f.close()
        print('found cached version')
        return trainz, trainz.shape[0]#trainx, trainy, trainz, testx, testy, testz
    except :
        print 'loading raw images' 
        data_path = home + "ift6266/inpainting"
        split="train2014"
        data_path = os.path.join(data_path, split)
        imgs = glob.glob(data_path + "/*.jpg")
        print data_path 
        # sample a few TODO : remove this 
        if sample : imgs = imgs[14000:18000]
        
        X, Y, Z = [], [], []
        for i, img_path in enumerate(imgs):
            try : 
                img = Image.open(img_path)
                img = np.array(img)
                if easy : 
                    if len(img.shape) != 3 : 
                        continue
                    Z.append(img)
            except :
                pass

        print str(len(Z)) + ' images found.'
        Image.fromarray(Z[-1]).show()
        Z = np.array(Z)
        Z = np.transpose(Z, axes=[0,3,1,2]).astype('float32')
        
        amt = Z.shape[0]
        idx1 = int(ds_split[0]*amt)
        idx2 = int((ds_split[0] + ds_split[1])*amt)
        
        f = file(home + name ,"wb")
        np.save(f,Z)

        return Z, Z.shape[0]    



def saveImage(imageData, path, side=8):

    # format data appropriately
    imageData = imageData.transpose(0,2,3,1).astype('uint8')

    #creates a new empty image, RGB mode, and size 400 by 400.
    new_im = Image.new('RGB', (64*side,64*side))
    
    index = 0
    for i in xrange(0,(side)*64,64):
        for j in xrange(0,(side)*64,64):
            #paste the image at location i,j:
            img = Image.fromarray(imageData[index])
            #img.show()
            new_im.paste(img, (i,j))
            index += 1

    new_im.save(path + '.png')

def optimizer_factory(optimizer, grads, params, eta):
    if optimizer == 'rmsprop' : 
        return lasagne.updates.rmsprop(
            grads, params, learning_rate=eta)
    elif optimizer == 'adam' : 
        return lasagne.updates.adam(
            grads, params, learning_rate=eta)
    else : 
        raise Exception(optimizer + ' not supported')

def format_imgs(samples, flatten=False):
    samples *= 0.5; samples += 0.5; samples *= 255.
    samples.astype('uint8')
    if len(samples.shape) == 5:
        if flatten: 
            samples = extract_video_frames(samples)
            samples = samples.transpose(0,2,3,1).astype('uint8')
        else :
            samples = samples.transpose(0,1,3,4,2).astype('uint8')
    else :
        samples = samples.transpose(0,2,3,1).astype('uint8')
    return samples



def data_iterator(batchsize):
    # step 1 : load the data 
    inputs, _ = load_dataset()
    inputs /= 255.
    inputs -= 0.5
    inputs /= 0.5
    while True : 
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt]



