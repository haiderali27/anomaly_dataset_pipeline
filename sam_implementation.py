import random
from threading import Event
import threading
import torch
from segment_anything import sam_model_registry
import supervision as sv
import os
from groundingdino.util.inference import Model
from typing import List
import cv2
from segment_anything import SamPredictor
import numpy as np
from queue import Queue
from typing import List
from itertools import cycle
from threading import Thread, Event
import time
import gc
#from torch.multiprocessing import Pool, Process, set_start_method



class SAMImplem:
    
    def __init__(self, USE_THREADS=False, THREAD_NUMBERS=2, IMAGE_POSTFIX="", USE_PREDEFINED_FILES=False, PRE_DEFINED_ARRAY_PATH='',CROP_MASK_IN_DIR=os.path.join(os.getcwd(), 'in_mask'), CROP_MASK_OUT_DIR=os.path.join(os.getcwd(), 'out_mask'), CROP_MASK_GREY=False, CROP_MASK=False, MASKS_COUNT= 2, MERGE_MASKS=False, USE_GAUSSIAN_BLUR=False, USE_MASK_BOUNDS=False, IN_DATA_PATH=os.path.join(os.getcwd(), 'in'), BREAK_IMAGE=False, IMAGE_OUTPUT_SIZE_RANGE = (768, 768), IMAGE_INPUT_SIZE_RANGE=(768, 768), OUT_DATA_PATH=os.path.join(os.getcwd(),'out'), CROP_BOUNDING_BOX = False, LOAD_ALL=False, DIVISONS=1, TOT_IMAGES=50, IN_RESIZE=True, IN_RESIZE_SIZE=(512, 512), selected_classes=['car', 'dog', 'person', 'nose', 'chair', 'shoe', 'ear'], box_threshold=0.35, text_threshold=0.25, LOAD_MODELS=True):
        
        if(LOAD_MODELS):
             gc.collect()
             torch.cuda.empty_cache()
             
             self.GROUNDING_DINO_CONFIG_PATH = os.path.join(os.getcwd(), "config/GroundingDINO_SwinT_OGC.py")
             self.GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(os.getcwd(), "weights/groundingdino_swint_ogc.pth")
             self.grounding_dino_model = Model(model_config_path=self.GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=self.GROUNDING_DINO_CHECKPOINT_PATH)
             self.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
             self.MODEL_TYPE = "vit_h"
             self.CHECKPOINT_PATH="weights/sam_vit_h_4b8939.pth"
             self.sam = sam_model_registry[self.MODEL_TYPE](checkpoint=self.CHECKPOINT_PATH).to(device=self.DEVICE)
             #self.sam.share_memory()
             self.sam_predictor=SamPredictor(self.sam)
            #self.sam_predictor.model.share_memory()
        self.IMAGE_POSTFIX = IMAGE_POSTFIX
        self.CROP_MASK_GREY=CROP_MASK_GREY
        self.CROP_MASK=CROP_MASK
        self.CROP_MASK_IN_DIR= CROP_MASK_IN_DIR
        self.CROP_MASK_OUT_DIR= CROP_MASK_OUT_DIR
        self.CLASSES = selected_classes
        self.BOX_TRESHOLD = box_threshold
        self.TEXT_TRESHOLD = text_threshold
        self.IN_DATA_PATH = IN_DATA_PATH
        self.OUT_DATA_PATH = OUT_DATA_PATH
        self.IN_RESIZE_SIZE = IN_RESIZE_SIZE
        self.THREAD_NUMBERS = THREAD_NUMBERS
        self.USE_THREADS = USE_THREADS
        self.IN_RESIZE = IN_RESIZE
        self.LOAD_ALL = LOAD_ALL
        self.TOT_IMAGES = TOT_IMAGES
        self.DIVISIONS = DIVISONS
        self.CROP_BOUNDING_BOX = CROP_BOUNDING_BOX
        self.IMAGE_INPUT_SIZE_RANGE = IMAGE_INPUT_SIZE_RANGE
        self.IMAGE_OUTPUT_SIZE_RANGE = IMAGE_OUTPUT_SIZE_RANGE
        self.BREAK_IMAGE = BREAK_IMAGE
        self.USE_MASK_BOUNDS=USE_MASK_BOUNDS
        self.USE_GAUSSIAN_BLUR = USE_GAUSSIAN_BLUR
        self.MERGE_MASKS = MERGE_MASKS
        self.MASKS_COUNT = MASKS_COUNT
        self.PRE_DEFINED_ARRAY_PATH = PRE_DEFINED_ARRAY_PATH
        self.USE_PREDEFINED_FILES = USE_PREDEFINED_FILES


    '''
    def process_context(self, job: SAMImplem) -> None:
        handle = multiprocessing.current_process()
        if job.do_log:
            print(f"starting {job.name} at {handle.name} at {datetime.now()}")
        time.sleep(1.0)
        if job.do_log:
            print(f"finished {job.name} at {handle.name} at {datetime.now()}")


    def execute_jobs_3(self, jobs: List[SAMImplem], processes: int = 2) -> None:
        with Pool(processes) as pool:
            pool.map(self.process_context, jobs)      
    '''

    def copy_mask(self, x1, y1, x2, y2 , image_name):
         image_name = image_name.replace(".jpg", ".png")
         image = cv2.imread(f'{self.CROP_MASK_IN_DIR}/{image_name}')
         image = image[y1:y2, x1:x2]
         if self.CROP_MASK_GREY:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Threshold the image to get the binary mask of the non-black regions
                ret, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

                # Invert the mask to get a mask for the black regions
                mask_inv = cv2.bitwise_not(mask)

                # Create a white image with the same size as the original image
                white_image = np.full_like(image, 255)

                # Copy the non-black regions from the original image to the white image using the mask
                result = cv2.bitwise_and(image, image, mask=mask_inv)

                # Add the white image to the result image using the mask
                result = cv2.add(result, white_image, mask=mask)

                filename, extension = os.path.splitext(image_name)
                new_filename = f"{filename}{self.IMAGE_POSTFIX}_mask{extension}"
                cv2.imwrite(f'{self.CROP_MASK_OUT_DIR}/{new_filename}', result)
                
              
         else:
            filename, extension = os.path.splitext(image_name)
            new_filename = f"{filename}{self.IMAGE_POSTFIX}_mask{extension}"
            cv2.imwrite(f'{self.CROP_MASK_OUT_DIR}/{new_filename}', image)
    

    def copy_segmentation(self, ref_directory_path, actual_directory ,output_directory, png_format=True):
        for img in os.listdir(ref_directory_path):
                imgg = f'{img}'
                image = None
                if png_format:   
                    imgg = f'{img}'.rsplit('.', 1)[0] + ".png"
                    image = cv2.imread(f'{actual_directory}/{imgg}')
                
                
                if image is None:
                            continue
                cv2.imwrite(f'{output_directory}/{imgg}', image)

    def convert_segmentation_to_white(self, ref_directory_path, actual_directory ,output_directory):
        for img in os.listdir(ref_directory_path):
                
                image = cv2.imread(f'{actual_directory}/{img}')
                                # Convert the image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Threshold the image to get the binary mask of the non-black regions
                ret, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

                # Invert the mask to get a mask for the black regions
                mask_inv = cv2.bitwise_not(mask)

                # Create a white image with the same size as the original image
                white_image = np.full_like(image, 255)

                # Copy the non-black regions from the original image to the white image using the mask
                result = cv2.bitwise_and(image, image, mask=mask_inv)

                # Add the white image to the result image using the mask
                result = cv2.add(result, white_image, mask=mask)
                if result is None:
                            continue
                cv2.imwrite(f'{output_directory}/{img}', result)


    def process_image(self, img):
                #handle = multiprocessing.current_process()
                image = cv2.imread(f'{self.IN_DATA_PATH}/{img}')
                #print('#####################################################',f'{self.IN_DATA_PATH}/{img}')
                height, width, _ = image.shape
                if height < self.IMAGE_INPUT_SIZE_RANGE[0] or width < self.IMAGE_INPUT_SIZE_RANGE[1]:
                    return

                if self.IN_RESIZE:
                    image = cv2.resize(image, self.IN_RESIZE_SIZE)
             
                result = self.clear_backgrounds(img, image)
                
                #print(len(result))
                if result is None:
                            return
                

                if self.BREAK_IMAGE:
                     for index, result_image in enumerate(result):
                        filename, extension = os.path.splitext(img)
                        new_filename = f"{filename}_{index}{self.IMAGE_POSTFIX}{extension}"
                        #print(new_filename)
                        if f'{filename}_{index}.jpg' in self.PREDEFINED_FILES_LIST_OUTPUT:
                              cv2.imwrite(f'{self.OUT_DATA_PATH}/{new_filename}', result_image)
                              #self.PREDEFINED_FILES_LIST_OUTPUT.remove(f'{filename}_{index}.jpg')
                else:
                    filename, extension = os.path.splitext(img)
                    new_filename = f"{filename}{self.IMAGE_POSTFIX}{extension}"
                    cv2.imwrite(f'{self.OUT_DATA_PATH}/{new_filename}', result)



         
    

    def load_images_and_clear_backgrounds(self):
        if self.USE_PREDEFINED_FILES:
             self.PREDEFINED_FILES_LIST = np.load(self.PRE_DEFINED_ARRAY_PATH)
             if self.BREAK_IMAGE: 
                  self.PREDEFINED_FILES_LIST_OUTPUT = self.PREDEFINED_FILES_LIST.copy().tolist()
                  self.PREDEFINED_FILES_LIST = self.PREDEFINED_FILES_LIST.tolist()
                  i = 0 
                  for s in self.PREDEFINED_FILES_LIST:
                    k = s.split('_')[0]+ '.'+s.split('_')[1].split('.')[1]
                    self.PREDEFINED_FILES_LIST[i] =  k
                    i+=1 
                  self.PREDEFINED_FILES_LIST =np.unique(np.array(self.PREDEFINED_FILES_LIST))
                  
                  self.PREDEFINED_FILES_LIST = self.PREDEFINED_FILES_LIST .tolist()

             
             if self.USE_THREADS is False:
                print('Not using threads')
                for img in self.PREDEFINED_FILES_LIST: 
                    self.process_image(img=img)      

             else:
                handles: List[Thread] = list()
                queues: List[Queue[str]] = list()
                stop_events: List[Event] = list()
                for i in range(2):
                    queue: Queue = Queue()
                    stop_event = Event()
                    handle = Thread(target=self.process_image_queue, name=f"thread {i}", args=(queue, stop_event))
                    handle.start()
                    handles.append(handle)
                    queues.append(queue)
                    stop_events.append(stop_event)
                pool = cycle(queues)
                while 0 < len(self.PREDEFINED_FILES_LIST):
                    job = self.PREDEFINED_FILES_LIST.pop()
                    queue = next(pool)
                    queue.put(job)
                # No more jobs to give
                for stop_event in stop_events:
                    stop_event.set()
                for handle in handles:
                    handle.join()   
             return 
             import torch.multiprocessing as mp
             mp.set_start_method('spawn')
            
            
             for img in self.PREDEFINED_FILES_LIST: 
                self.process_image(img=img)
                
                #with Pool(4) as pool:
                #  pool.map(self.process_image, [img])
                

                  
                  
             

        if self.LOAD_ALL:
            for img in os.listdir(self.IN_DATA_PATH):
                
                image = cv2.imread(f'{self.IN_DATA_PATH}/{img}')
                height, width, _ = image.shape
                if height < self.IMAGE_INPUT_SIZE_RANGE[0] or width < self.IMAGE_INPUT_SIZE_RANGE[1]:
                    continue

                if self.IN_RESIZE:
                    image = cv2.resize(image, self.IN_RESIZE_SIZE)
             
                result = self.clear_backgrounds(img, image)
                
                #print(len(result))
                if result is None:
                            continue
                

                if self.BREAK_IMAGE:
                     for index, result_image in enumerate(result):
                        filename, extension = os.path.splitext(img)
                        new_filename = f"{filename}_{index}{self.IMAGE_POSTFIX}{extension}"
                        cv2.imwrite(f'{self.OUT_DATA_PATH}/{new_filename}', result_image)
                else:
                    filename, extension = os.path.splitext(img)
                    new_filename = f"{filename}{self.IMAGE_POSTFIX}{extension}"
                    cv2.imwrite(f'{self.OUT_DATA_PATH}/{new_filename}', result)


                          
                     
                     
        else:
            number_of_images = len(os.listdir(self.IN_DATA_PATH))
            #print(number_of_images, '############', imagePath, os.listdir(imagePath) )
            if(self.DIVISIONS==1):
                
                data_start = 0 if self.TOT_IMAGES > number_of_images  else random.randint(0, number_of_images-self.TOT_IMAGES)
                data_end = number_of_images if self.TOT_IMAGES > number_of_images else  data_start+self.TOT_IMAGES
                #print(f'dataStart: {data_start}, dataEnd:{data_end}')
                i = 0
                for img in os.listdir(self.IN_DATA_PATH):
                    if(i<data_start):
                        i+=1
                        continue;
                    if (i>=data_end):
                        break;
                    image = cv2.imread(f'{self.IN_DATA_PATH}/{img}')
                    height, width, _ = image.shape

                    if height < self.IMAGE_INPUT_SIZE_RANGE[0] or width < self.IMAGE_INPUT_SIZE_RANGE[1]:
                        continue
                    if self.IN_RESIZE:
                            image = cv2.resize(image, self.IN_RESIZE_SIZE)
                    result = self.clear_backgrounds(img, image)

                    if result is None:
                            continue

                    #if self.BREAK_IMAGE:
                    #    for index, result_image in enumerate(result):
                    #        cv2.imwrite(f'{self.OUT_DATA_PATH}/{img}_{index}{self.IMAGE_POSTFIX}', result_image)
                    #else:
                    #    cv2.imwrite(f'{self.OUT_DATA_PATH}/{img}{self.IMAGE_POSTFIX}', result)

                    if self.BREAK_IMAGE:
                     for index, result_image in enumerate(result):
                        filename, extension = os.path.splitext(img)
                        new_filename = f"{filename}_{index}{self.IMAGE_POSTFIX}{extension}"
                        cv2.imwrite(f'{self.OUT_DATA_PATH}/{new_filename}', result_image)
                    else:
                        filename, extension = os.path.splitext(img)
                        new_filename = f"{filename}{self.IMAGE_POSTFIX}{extension}"
                        cv2.imwrite(f'{self.OUT_DATA_PATH}/{new_filename}', result)



                    i+=1
            else:
                data_divisions = number_of_images/self.DIVISIONS
                data_divisions = int(data_divisions)
                dataSizes = self.TOT_IMAGES/self.DIVISIONS
                dataSizes = int(dataSizes)
                for chunk in range(self.DIVISIONS):  
                    data_start = random.randint((chunk*data_divisions), ((chunk+1)*data_divisions)-dataSizes)
                    data_end = data_start+dataSizes
                    #print(f'dataSizes:{dataSize}, data_start:{data_start}, data_end:{data_end}')
                    j = 0
                    for img in os.listdir(self.IN_DATA_PATH):
                        if(j<data_start):
                            j+=1
                            continue;
                        if (j>=data_end):

                            break;
                        image = cv2.imread(f'{self.IN_DATA_PATH}/{img}')
                        height, width, _ = image.shape

                        if height < self.IMAGE_INPUT_SIZE_RANGE[0] or width < self.IMAGE_INPUT_SIZE_RANGE[1]:
                            continue
                        if self.IN_RESIZE:
                            image = cv2.resize(image, self.IN_RESIZE_SIZE)
                        result = self.clear_backgrounds(img, image)
                        if result is None:
                            continue

                        #if self.BREAK_IMAGE:
                        #    for index, result_image in enumerate(result):
                        #        cv2.imwrite(f'{self.OUT_DATA_PATH}/{img}_{index}{self.IMAGE_POSTFIX}', result_image)
                        #else:
                        #    cv2.imwrite(f'{self.OUT_DATA_PATH}/{img}{self.IMAGE_POSTFIX}', result)


                       
                        if self.BREAK_IMAGE:
                            for index, result_image in enumerate(result):
                                filename, extension = os.path.splitext(img)
                                new_filename = f"{filename}_{index}{self.IMAGE_POSTFIX}{extension}"
                                cv2.imwrite(f'{self.OUT_DATA_PATH}/{new_filename}', result_image)
                        else:
                            filename, extension = os.path.splitext(img)
                            new_filename = f"{filename}{self.IMAGE_POSTFIX}{extension}"
                            cv2.imwrite(f'{self.OUT_DATA_PATH}/{new_filename}', result)     


                        j+=1
        


    def process_image_queue(self, queue: Queue, stop_event: Event):
        handle = threading.current_thread()
        do_continue = True
        while do_continue:
            if queue.empty():
                if stop_event.is_set():
                    do_continue = False
                else:
                    time.sleep(0.1)
            else:
                img = queue.get()
                while True:    
                    try:    
                        print(img, '############################################', f"starting {handle.name}")
                        #handle = multiprocessing.current_process()
                        image = cv2.imread(f'{self.IN_DATA_PATH}/{img}')
                        #print('#####################################################',f'{self.IN_DATA_PATH}/{img}')
                        height, width, _ = image.shape
                        if height < self.IMAGE_INPUT_SIZE_RANGE[0] or width < self.IMAGE_INPUT_SIZE_RANGE[1]:
                            return

                        if self.IN_RESIZE:
                            image = cv2.resize(image, self.IN_RESIZE_SIZE)
                    
                        result = self.clear_backgrounds(img, image)
                        
                        #print(len(result))
                        if result is None:
                                    return
                        

                        if self.BREAK_IMAGE:
                            for index, result_image in enumerate(result):
                                filename, extension = os.path.splitext(img)
                                new_filename = f"{filename}_{index}{self.IMAGE_POSTFIX}{extension}"
                                #print(new_filename)
                                if f'{filename}_{index}.jpg' in self.PREDEFINED_FILES_LIST_OUTPUT:
                                    cv2.imwrite(f'{self.OUT_DATA_PATH}/{new_filename}', result_image)
                                    #self.PREDEFINED_FILES_LIST_OUTPUT.remove(f'{filename}_{index}.jpg')
                        else:
                            filename, extension = os.path.splitext(img)
                            new_filename = f"{filename}{self.IMAGE_POSTFIX}{extension}"
                            cv2.imwrite(f'{self.OUT_DATA_PATH}/{new_filename}', result)
                        break
                    except Exception as e:
                        print(e)
                     
    def clear_backgrounds(self, image_name, image):
       
        detections = self.dino_detections(image)
        if self.USE_GAUSSIAN_BLUR:
             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
             b, g, r = cv2.split(image)

            # Apply a GaussianBlur to each color channel tried(5, 5) (3, 3) (1, 1)
             b_blur = cv2.GaussianBlur(b, (1, 1), 0)
             g_blur = cv2.GaussianBlur(g, (1, 1), 0)
             r_blur = cv2.GaussianBlur(r, (1, 1), 0)

            # Apply the Laplacian filter to each color channel
             b_laplacian = cv2.Laplacian(b_blur, cv2.CV_64F)
             g_laplacian = cv2.Laplacian(g_blur, cv2.CV_64F)
             r_laplacian = cv2.Laplacian(r_blur, cv2.CV_64F)

            # Convert the Laplacian results back to uint8
             b_laplacian = np.uint8(np.absolute(b_laplacian))
             g_laplacian = np.uint8(np.absolute(g_laplacian))
             r_laplacian = np.uint8(np.absolute(r_laplacian))

            # Merge the color channels back together
             sharpened = cv2.merge([b_laplacian, g_laplacian, r_laplacian])

             contrast_factor = 10 # You can adjust this value according to your preference

            # Split the image into its color channels (B, G, R)
             b, g, r = cv2.split(sharpened)

            # Apply the contrast adjustment to each channel
             adjusted_b = np.clip(contrast_factor * b, 0, 255).astype(np.uint8)
             adjusted_g = np.clip(contrast_factor * g, 0, 255).astype(np.uint8)
             adjusted_r = np.clip(contrast_factor * r, 0, 255).astype(np.uint8)

            # Merge the adjusted channels back into an RGB image
             adjusted_image = cv2.merge([adjusted_b, adjusted_g, adjusted_r])
             image_copy = image
             image = adjusted_image


        #detections = self.dino_detections(image)

        if self.CROP_BOUNDING_BOX: 
            if len(detections.xyxy) ==0:
                 return None
            x1, y1, x2, y2 = map(int, detections.xyxy[0])
            # # Crop the image
            cropped_image = image[y1:y2, x1:x2]
            height, width, _ = cropped_image.shape
            if height < self.IMAGE_OUTPUT_SIZE_RANGE[0] or width < self.IMAGE_OUTPUT_SIZE_RANGE[1]:
                return None
            else:
                 if self.CROP_MASK:
                      self.copy_mask(x1, y1, x2, y2, image_name=image_name)
                      
                 
             
        detections.mask, scores = self.sam_detections(image, detections=detections, xyxy=detections.xyxy[0])
        #print(f'image_name:{image_name}, scores: {scores}')
        if len(detections.mask)==0:
            return None
        mask = detections.mask[0]

        if self.USE_MASK_BOUNDS:
             mask = mask.astype(np.uint8) * 255
             mask = mask[:, :, np.newaxis]
             mask = np.repeat(mask, 3, axis=2)

             threshold = 127  # You can adjust this threshold based on your image characteristics
             binary_mask = (mask > threshold) * 255
             
             #print('##############################',binary_mask.shape, mask.shape)
             #print(binary_mask)

             # If you want to ensure the result is of integer type (0 and 255)
             binary_mask = binary_mask.astype(np.uint8)
             binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)  # Convert to single-channel

            # Find contours in the binary mask
             contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a blank image to draw contours
             contour_image = np.zeros_like(binary_mask)

            # Draw the contours on the blank image
             cv2.drawContours(contour_image, contours, -1, (255), thickness=cv2.FILLED)
             if self.USE_GAUSSIAN_BLUR:
                  image = image_copy

             true_indices = np.argwhere(contour_image)
             indexes=[]
             for index in true_indices:
                indexes.append([index[0], index[1]])
                #print(tuple(index))
                #print(indexes)
             indexes=np.array(indexes)

             true_indices=indexes
             mask = np.zeros_like(image, dtype=np.uint8)

             # Set the regions specified by the true indices to white
             mask[true_indices[:, 0], true_indices[:, 1]] = [255, 255, 255]

             # Create the result image by masking the original image
             result_image = cv2.bitwise_and(image, mask)
             if self.CROP_BOUNDING_BOX: 
                x1, y1, x2, y2 = map(int, detections.xyxy[0])
                # # Crop the image
                result_image = result_image[y1:y2, x1:x2]
                if result_image is not None:
                            height, width, _ = result_image.shape
                if self.BREAK_IMAGE:
                        half_height = height/2
                        half_width= width/2
                        forth_height = height/4
                        forth_width= width/4
                        if (half_height < self.IMAGE_OUTPUT_SIZE_RANGE[0] or half_width < self.IMAGE_OUTPUT_SIZE_RANGE[1]):
                            #random_number = random.choice([1, 2, 4])
                            return self.split_image(result_image, 1)
                        else:
                            if (forth_height < self.IMAGE_OUTPUT_SIZE_RANGE[0] or forth_width < self.IMAGE_OUTPUT_SIZE_RANGE[1]):
                                 return self.split_image(result_image, 4)
                            
                            return self.split_image(result_image, 2)



        if self.USE_GAUSSIAN_BLUR:
            image = image_copy



        true_indices = np.argwhere(mask)
        indexes=[]
        for index in true_indices:
            indexes.append([index[0], index[1]])
            #print(tuple(index))
            #print(indexes)
        indexes=np.array(indexes)

        true_indices=indexes
        mask = np.zeros_like(image, dtype=np.uint8)

        # Set the regions specified by the true indices to white
        mask[true_indices[:, 0], true_indices[:, 1]] = [255, 255, 255]

        # Create the result image by masking the original image
        result_image = cv2.bitwise_and(image, mask)
        if self.CROP_BOUNDING_BOX: 
            x1, y1, x2, y2 = map(int, detections.xyxy[0])
            # # Crop the image
            result_image = result_image[y1:y2, x1:x2]
            if result_image is not None:
                            height, width, _ = result_image.shape
            if self.BREAK_IMAGE:
                        half_height = height/2
                        half_width= width/2
                        forth_height = height/4
                        forth_width= width/4
                        if (half_height < self.IMAGE_OUTPUT_SIZE_RANGE[0] or half_width < self.IMAGE_OUTPUT_SIZE_RANGE[1]):
                            #random_number = random.choice([1, 2, 4])
                            return self.split_image(result_image, 1)
                        else:
                            if (forth_height < self.IMAGE_OUTPUT_SIZE_RANGE[0] or forth_width < self.IMAGE_OUTPUT_SIZE_RANGE[1]):
                                 return self.split_image(result_image, 4)
                            
                            return self.split_image(result_image, 2)


        return result_image
    

# Function to split the image based on the specified number of parts
    def split_image(self, image, num_parts):
        height, width, _ = image.shape
        if num_parts ==1:
             return [image]
        if num_parts == 2:
            # Split into 2 squares
            half_size = min(height, width) // 2
            part1 = image[:half_size, :]
            part2 = image[half_size:2*half_size, :]
            return [part1, part2]

        elif num_parts == 3:
            # Split into 2 squares and 1 rectangle
            square_size = min(height, width) // 2
            part1 = image[:square_size, :]
            part2 = image[square_size:2*square_size, :]
            part3 = image[2*square_size:, :]
            return [part1, part2, part3]

        elif num_parts == 4:
            # Split into 4 parts
            half_height = height // 2
            half_width = width // 2
            part1 = image[:half_height, :half_width]
            part2 = image[:half_height, half_width:]
            part3 = image[half_height:, :half_width]
            part4 = image[half_height:, half_width:]
            return [part1, part2, part3, part4]

        else:
            raise ValueError("Invalid number of parts")


    def dino_detections(self, image):
        detections = self.grounding_dino_model.predict_with_classes(
        image=image,
        classes=self.enhance_class_name(class_names=self.CLASSES),
        box_threshold=self.BOX_TRESHOLD,
        text_threshold=self.TEXT_TRESHOLD
        )
        return detections
    def sam_detections(self, image, detections, xyxy):
        detections.mask, scores = self.segment(
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=[xyxy]
        )
        return detections.mask, scores

    
    def box_annotator(self, image, detections, labels=None):
        box_annotator = sv.BoxAnnotator()
        if labels==None:
            labels = [
                f"{self.CLASSES[class_id]} {confidence:0.2f}" 
                for _, _, confidence, class_id, _ 
                in detections]
        
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        return annotated_frame
    def mask_annotator(self, image, detections):
        mask_annotator = sv.MaskAnnotator()
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        return annotated_image



    def enhance_class_name(self, class_names: List[str]) -> List[str]:
        return [
            f"all {class_name}s"
            for class_name
            in class_names
        ]
    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.sam_predictor.set_image(image=image, image_format="RGB")
        result_masks = []
        masks_sorted = []

        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index_sorted = np.argsort(scores)
            for ind in index_sorted:
                masks_sorted.append(masks[ind])
            index = np.argmax(scores)
            result_masks.append(masks[index])

        if(self.MERGE_MASKS):
          masks_sorted = np.array(masks_sorted)
          if(self.MASKS_COUNT>len(masks_sorted)):
                result_array = np.logical_or.reduce(masks_sorted)
                result_array = np.expand_dims(result_array, axis=0)
                return result_array, -np.sort(-scores)
          else: 
                result_array = np.logical_or.reduce(masks_sorted[-self.MASKS_COUNT:])
                result_array = np.expand_dims(result_array, axis=0)
                return result_array, -np.sort(-scores)
               
         
        return np.array(result_masks)
