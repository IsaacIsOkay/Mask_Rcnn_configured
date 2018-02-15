from config import Config
import utils
import os 
import numpy as np
from keras import backend as K
from PIL import Image
from keras.preprocessing.image import *
import model as modellib
import random
import visualize
from visualize import display_images
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import log
import time
%matplotlib
# Root directory of the project
ROOT_DIR = os.getcwd()

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

PLANT = "broccoli"

MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_%s.h5"% (PLANT))

class weedSpecConfig(Config):
	# Give the configuration a recognizable name
    NAME = "WeedSpec"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 2  # background and lettuce

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1


class WeedSpecDataset(utils.Dataset):

	def load_plants(self, dataset_dir, data_id_file):
		
		self.add_class("plants", 1, PLANT)
		img_dir = os.path.join(dataset_dir, "image")
		train_path = os.path.join(dataset_dir, data_id_file)

		image_ids = []
		fp = open(train_path)
		lines = fp.readlines()
		fp.close()
		for line in lines:
			line = line.strip('\n')
			image_ids.append(line)
		for i in image_ids:
			self.add_image("plants", image_id=i, 
			path=os.path.join(img_dir,'%s.png'%(i)),
			data_id_file=data_id_file,
			data_dir = dataset_dir)

	def load_mask(self, image_id):
		
		#label_path = "/home/default/.keras/datasets/weedspic/lettuce/train.txt"
		info = self.image_info[image_id]
		data_dir = info["data_dir"]
		label_path = os.path.join(data_dir, info["data_id_file"])
		label_dir = "/home/default/.keras/datasets/weedspic/%s/label"%(PLANT)
		#print(label_dir)
		
		label_ids = []
		fp = open(label_path)
		lines = fp.readlines()
		fp.close()
		for line in lines:
			line = line.strip('\n')
			label_ids.append(line)
		
		label_file = ("%s.png"%(label_ids[image_id]))
		label_filepath = os.path.join(label_dir, label_file)
		
		label = Image.open(label_filepath)
		mask = img_to_array(label)
		mask[np.where(mask == 255)] = 1
		class_ids = np.array([1,])
		return mask, class_ids.astype(np.int32)	
	
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		
		if info["source"] == "plants":
			return info
		else:
			super(self.__class__).image_reference(self,image_id)


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax




inference_config = weedSpecConfig()		

debug_save_dir = "/home/default/KerasWorkspace/maskrcnn/Mask_RCNN-2.0/debug/"
data_dir = "/home/default/.keras/datasets/weedspic/%s"%(PLANT)

dataset_val = WeedSpecDataset()
dataset_val.load_plants(data_dir, "all_val.txt")
dataset_val.prepare()

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

model_path = os.path.join(MODEL_DIR, "mask_rcnn_%s.h5"%PLANT)

assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


'''
start_time = time.time()


duration = time.time() - start_time
'''
path = "/home/default/.keras/datasets/weedspic/lettuce/all_val.txt"
fp = open(path)

lines = fp.readlines()
fp.close()

all_nums = np.zeros(1180)

for line in lines:
	line = line.strip('\n')
	if(line != "0000"):
		line = line.lstrip('0')
		line = int(line)
		all_nums[line] = 1
imageCount = 0
accSum = 0
avgAccuracy = 0
for i in range(0, 720):
	
	##Phase masks together##
	if(all_nums[i] == 1):
		fileName = str(i).zfill(4)
		label_filepath = "/home/default/.keras/datasets/weedspic/broccoli/label/%s.png"%(dataset_val.image_reference(i)["id"])
		phased_mask = np.zeros((512,512))
		image_id = i
		
		original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
	   			modellib.load_image_gt(dataset_val, inference_config, 
		                   			image_id, use_mini_mask=False)
		
		results = model.detect([original_image], verbose=0)
		r = results[0]
		msk = r['masks']
		if(msk.size != 0):
			for m in np.rollaxis(msk, 2):
				for i in range(0,512):
					for k in range(0,512):				
						if(m[i][k] == 1):
							phased_mask[i][k] = 1

		label = Image.open(label_filepath)
		label = img_to_array(label.resize((512, 512), Image.NEAREST)).astype(np.uint8)
		label[np.where(label == 255)] = 1
		label = label[:,:,0]
		correct = 0
		cp = 0
		tp = 0
		total = 512 * 512
		pimgCount = 0
		paccSum = 0
		for a in range(0, 512):
			for b in range(0,512):
				if(phased_mask[a][b] == label[a][b]):
					correct += 1
				if(phased_mask[a][b] == 1):
					tp += 1
					if(phased_mask[a][b] == label[a][b]):
						cp += 1
		accuracy = (float)(correct/total)
		print("id: %d"%image_id,"accuracy: %f"%accuracy, "totalP: %d"%tp,"correctP: %d"%cp)
		imageCount += 1
		accSum += accuracy
		if(tp > 0):
			acc = (float)(cp/tp)
			print("PixelACC", image_id, acc)
			pimgCount += 1
			paccSum += acc
		#np.savetxt('%s%s(0).csv'%(debug_save_dir,"label_check"), label, delimiter=",")	
avgAccuracy = (float)(accSum/imageCount)
pavgAccuracy = (float)(paccSum/pimgCount)

print(avgAccuracy, "lettuce only accuracy", pavgAccuracy)

'''
res_1 = msk[:,:,0]
res_2 = msk[:,:,1]
#res_2 = res[:,:,1]
np.savetxt('%s%s(0).csv'%(debug_save_dir,"mask_eval"), res_1, delimiter=",")
np.savetxt('%s%s(0).csv'%(debug_save_dir,"mask_eval_2"), res_2, delimiter=",")
'''


