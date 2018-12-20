import os
import time
import numpy as np
import cv2
import json
from config import Config
import utils
import model as modellib

# Root directory of the project
ROOT_DIR = os.getcwd()

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

PLANT = "Sorghum"

MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_%s.h5"% (PLANT))

LOAD_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_lettuce.h5")

DRIVE = "phenobot"
class GroundTruth():
    
    def __init__(self, polyId, img_height, img_width):
        self.next = None
        self.mask = np.zeros((img_height,img_width), dtype = "uint8")
        self.max_x = 0
        self.max_y = 0
        self.min_x = 100000
        self.min_y = 100000
        self.p_Id = polyId
    
    def get_lists(self):
        
        maskList = []
        bboxList = []
        if(self.next == None):
            maskList.append(self.mask)
            bboxList.append(np.array([self.min_y, self.min_x, self.max_y, self.max_x]))
            return maskList, bboxList
        else:
            maskList, bboxList = self.next.get_lists()
            maskList.append(self.mask)
            bboxList.append(np.array([self.min_y, self.min_x, self.max_y, self.max_x]))
            return maskList, bboxList
    def __str__(self):
        
        temp = "____\nMax_x: " + str(self.max_x) + ", Min_x: " + str(self.min_x)
        temp = temp + "\n\nMax_y: " + str(self.max_y) + ", Min_y: " + str(self.min_y)

        return temp;

class StalkSpecConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "StalkSpec"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 2  # Background and sorghum

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
    STEPS_PER_EPOCH = 206

class StalkSpecDataset(utils.Dataset):
    
    """Load the dataset.
        dataset_dir: The root directory of the dataset.
        class_ids: If provided, only loads images that have the given classes.
    """
    def load_stalks(self, dataset_dir, data_id_file):
       
        self.add_class("stalk", 1, PLANT)
        # Path
        image_dir = os.path.join(dataset_dir, "images")
        train_path = os.path.join(dataset_dir, data_id_file)


        image_ids = []
        labels = []
        paths = []
        #fp = filePaths
        fp = open(train_path)
        lines = fp.readlines()
        fp.close()
        for line in lines:
            line = line.strip('\n')
            image_ids.append(int(line))
            json_file = open(dataset_dir + "/labels/%s.json"%(line))
            json_str = json_file.read()
            if(len(json_str) == 0): print(line)
            json_data = json.loads(json_str)
            labels.append(json_data)
            paths.append(line)
        pth = os.path.join(image_dir, "0003.jpg")
        img = cv2.imread(pth, 0)
        h = img.shape[0]
        w = img.shape[1]      
        for ix, i in enumerate(image_ids):
            self.add_image("stalk", image_id=i,
                           path=os.path.join(image_dir, "%s.jpg" % paths[ix]),
                           data_id_file=data_id_file,
                           data_dir = dataset_dir,
                           annotations = labels[ix],
                           height = h,
                           width = w)
        
    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        bboxes: array [num_instances, (y1, x1, y2, x2)].
        """
        image_info = self.image_info[image_id]
    
        json_data = image_info['annotations']

        img_scale = 5
        
        img_height = image_info['height']
        img_width = image_info['width']
        
        
        masks = []
        bboxes = []
        groundTruths = []

        

        for poly in json_data:
            polyId = poly['polygon_id']
            p = int(polyId)
            points = []
            currGt = None
            #find the ground truth        
            for gt in groundTruths:              
                if(gt.p_Id == p):
                    currGt = gt
                    break
            #if no ground truth was found make a new one and add it to the list
            if(currGt == None):
                currGt = GroundTruth(p, img_height, img_width)
                if(len(groundTruths) > 0):
                    groundTruths[len(groundTruths)-1].next = currGt
                groundTruths.append(currGt)
            #update the mask and the min/max x and y
            for vertex in poly['vertices']:
                x = int((int(vertex['x']) * img_scale)/2)
                y = int((int(vertex['y']) * img_scale)/2)
                points.append([x,y]) 
                if(x > currGt.max_x): currGt.max_x = x
                if(x < currGt.min_x): currGt.min_x = x
                if(y > currGt.max_y): currGt.max_y = y
                if(y < currGt.min_y): currGt.min_y = y
            points = np.array(points)
            mask = currGt.mask
            cv2.fillConvexPoly(mask, points, 255)
 
        masks, bboxs = groundTruths[0].get_lists()
        bboxs = np.asarray(bboxs)
        masks = np.asarray(masks)
        masks = np.swapaxes(masks, 0, 2)
        masks = np.swapaxes(masks, 0, 1)

        class_ids = []
        for i in range(0, masks.shape[2]):
            class_ids.append(1)
        
        class_ids = np.array(class_ids)

        return masks, class_ids.astype(np.int32), bboxs
    
        
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        
        if info["source"] == "plants":
            return info


config = StalkSpecConfig()        

debug_save_dir = "/media/%s/ISAACTEGLER/maskrcnn/Mask_RCNN-2.0/debug/" %(DRIVE)
data_dir = "/media/%s/ISAACTEGLER/data_sets/%s/squashed"%(DRIVE, PLANT)

dataset_train = StalkSpecDataset()
dataset_train.load_stalks(data_dir, "train.txt")
print(dataset_train.image_reference(3))
dataset_train.prepare()

dataset_val = StalkSpecDataset()
dataset_val.load_stalks(data_dir, "val.txt")
dataset_val.prepare()

model = modellib.MaskRCNN(mode="training", config = config, model_dir=MODEL_DIR)

model.load_weights(LOAD_MODEL_PATH, by_name=True,
		exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=30, 
            layers='all')
model_path = os.path.join(MODEL_DIR, "mask_rcnn_Sorghum_trained.h5")
model.keras_model.save_weights(model_path)



