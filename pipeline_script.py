import urllib.request

urls = [
    'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth',
    'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
]

for url in urls:
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, 'weights/'+filename)

print('weights downloaded')



import os
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
import gdown
import zipfile

def downloadFiles(in_file, out_dir):
    names_array, download_urls = getFilesAndFiles(in_file)
    headers = {
    'User-Agent': 'My User Agent 1.0'
    }
    i = 0
    for name in names_array:
        img_data = requests.get(download_urls[i], headers=headers, stream=True).content
        with open(out_dir+'/'+name, 'wb') as handler:
            handler.write(img_data)
        i+=1

    
def getFilesAndFiles(in_file):
    f = open(in_file, "r")

    content = f.read()
    #print(content)

    #content_with_out_lines = "".join([s for s in content.strip().splitlines(True) if s.strip()])
    #print(content_with_out_lines)

    names_array = []
    download_urls = []
    i = 1 
    for s in content.splitlines(True):
        if(s.strip()):
            #print(i, j)
            #print(s, '##########',i)
            if i%3==1:
                names_array.append(s.rstrip())
           
            if i%3==0:
                download_urls.append(s.rstrip())
            i+=1
        
    return names_array, download_urls

download_dir = os.getcwd()+'/cars'
os.makedirs(download_dir, exist_ok=True)
web_train = os.getcwd()+'/cars/web_train'
web_test = os.getcwd()+'/cars/web_test' 
os.makedirs(web_train, exist_ok=True)
os.makedirs(web_test, exist_ok=True)
downloadFiles('assets/web_train.txt', os.getcwd()+'/cars/web_train')
downloadFiles('assets/web_test.txt', web_test)
print(f'Web Datasets Places successFully')

api = KaggleApi()
api.authenticate()  

dataset_names = ['jessicali9530/stanford-cars-dataset', 'humansintheloop/car-parts-and-car-damages', 'hamedahangari/internal-and-external-parts-of-cars']

for dataset_name in dataset_names:
    api.dataset_download_files(dataset_name, path=download_dir, unzip=True)
    print(f"Dataset '{dataset_name}' downloaded successfully.")



cardd_url = 'https://drive.google.com/uc?id=1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx'
#cardd_url='1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx'
output = os.getcwd()+'/cars/cardd.zip'
gdown.download(cardd_url, output, quiet=False)
extract_dir =  os.getcwd()+'/cars/'


with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

os.remove(output)



import sam_implementation 
import numpy as np

    


IN_DATA_PATH = os.getcwd()+'/cars/cars_test/cars_test'
OUT_DATA_PATH= os.getcwd()+'/dataset/cars/train/good'

os.makedirs(OUT_DATA_PATH, exist_ok=True)


sam_impl = sam_implementation.SAMImplem(USE_THREADS=False, THREAD_NUMBERS=2, USE_PREDEFINED_FILES=True, PRE_DEFINED_ARRAY_PATH='assets/standford_te.npy', IMAGE_POSTFIX= "_sf_te", MERGE_MASKS=True, CROP_BOUNDING_BOX=True , BREAK_IMAGE=True, IMAGE_OUTPUT_SIZE_RANGE=(512, 512), IMAGE_INPUT_SIZE_RANGE = (512, 512), IN_DATA_PATH=IN_DATA_PATH, OUT_DATA_PATH=OUT_DATA_PATH,IN_RESIZE=False,LOAD_ALL=True)
sam_impl.LOAD_ALL=False


images = sam_impl.load_images_and_clear_backgrounds()
print('####STAND_TEST_SEGMENTED')


sam_impl.IMAGE_POSTFIX= "_sf_tr"
sam_impl.IN_DATA_PATH = os.getcwd()+'/cars/cars_train/cars_train'
sam_impl.PRE_DEFINED_ARRAY_PATH='assets/standford_tr.npy'
images = sam_impl.load_images_and_clear_backgrounds()
print('####STAND_TRAIN_SEGMENTED')


sam_impl.BREAK_IMAGE=False

sam_impl.IMAGE_POSTFIX= "_eai"
sam_impl.PRE_DEFINED_ARRAY_PATH='assets/carparts_eai.npy'
sam_impl.IN_DATA_PATH = os.getcwd()+'/cars/Car parts/External/Air intake'
images = sam_impl.load_images_and_clear_backgrounds()

sam_impl.IMAGE_POSTFIX= "_efl"
sam_impl.PRE_DEFINED_ARRAY_PATH='assets/carparts_efl.npy'

sam_impl.IN_DATA_PATH = os.getcwd()+'/cars/Car parts/External/Fog light'
images = sam_impl.load_images_and_clear_backgrounds()

sam_impl.IMAGE_POSTFIX= "_ehl"
sam_impl.PRE_DEFINED_ARRAY_PATH='assets/carparts_ehl.npy'
sam_impl.IN_DATA_PATH = os.getcwd()+'/cars/Car parts/External/Headlight'
images = sam_impl.load_images_and_clear_backgrounds()

sam_impl.IMAGE_POSTFIX= "_etl"
sam_impl.PRE_DEFINED_ARRAY_PATH='assets/carparts_etl.npy'
sam_impl.IN_DATA_PATH = os.getcwd()+'/cars/Car parts/External/Tail light'
images = sam_impl.load_images_and_clear_backgrounds()
print('####CAR_PARTS_SEGMENTED')



sam_impl.IMAGE_POSTFIX= ""
sam_impl.USE_PREDEFINED_FILES= False
sam_impl.IN_DATA_PATH = os.getcwd()+'/cars/web_train'
images = sam_impl.load_images_and_clear_backgrounds()



np_array = np.load('assets/test_good.npy')

source_folder =os.getcwd()+'/dataset/cars/train/good'


# Destination folder
destination_folder = os.getcwd()+'/dataset/cars/test/good'

os.makedirs(destination_folder, exist_ok=True)

for file_name in np_array:
    # Create full paths for source and destination
    source_path = f"{source_folder}/{file_name}"
    destination_path = f"{destination_folder}/{file_name}"
    
    # Move the file from source to destination
    os.rename(source_path, destination_path)

sam_impl.IMAGE_POSTFIX= ""
sam_impl.USE_PREDEFINED_FILES= False
sam_impl.IN_DATA_PATH = os.getcwd()+'/cars/web_test'
sam_impl.OUT_DATA_PATH= os.getcwd()+'/dataset/cars/test/good'
images = sam_impl.load_images_and_clear_backgrounds()


sam_impl.USE_PREDEFINED_FILES= True
sam_impl.BREAK_IMAGE=False
sam_impl.PRE_DEFINED_ARRAY_PATH='assets/damage_cars.npy'
sam_impl.IMAGE_POSTFIX= ""
sam_impl.CROP_MASK=True
sam_impl.CROP_MASK_GREY=True
sam_impl.CROP_MASK_IN_DIR=os.getcwd()+'/cars/Car parts dataset/File1/masks_machine/'
sam_impl.CROP_MASK_OUT_DIR= os.getcwd()+'/dataset/cars/ground_truth/damage'
os.makedirs(sam_impl.CROP_MASK_OUT_DIR, exist_ok=True)
sam_impl.IN_DATA_PATH= os.getcwd()+'/cars/Car parts dataset/File1/img'
sam_impl.OUT_DATA_PATH= os.getcwd()+'/dataset/cars/test/damage'
os.makedirs(sam_impl.OUT_DATA_PATH, exist_ok=True)
sam_impl.load_images_and_clear_backgrounds()
print('####CAR_TEST_DAMAGE_SEGMENTED')





sam_impl.PRE_DEFINED_ARRAY_PATH='assets/test_damaged_dd_tr.npy'
sam_impl.CROP_MASK_GREY=False
sam_impl.CROP_MASK=True
sam_impl.CROP_MASK_IN_DIR=os.getcwd()+'/cars/CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Mask'
sam_impl.IN_DATA_PATH= os.getcwd()+'/cars/CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Image'
sam_impl.CROP_MASK_OUT_DIR= os.getcwd()+'/dataset/cars/ground_truth/damagedd'
os.makedirs(sam_impl.CROP_MASK_OUT_DIR, exist_ok=True)
sam_impl.OUT_DATA_PATH= os.getcwd()+'/dataset/cars/test/damagedd'
os.makedirs(sam_impl.OUT_DATA_PATH, exist_ok=True)
sam_impl.load_images_and_clear_backgrounds()



sam_impl.PRE_DEFINED_ARRAY_PATH='assets/test_damaged_dd_te.npy'
sam_impl.CROP_MASK_IN_DIR=os.getcwd()+'/cars/CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Mask'
sam_impl.IN_DATA_PATH= os.getcwd()+'/cars/CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Image'
sam_impl.load_images_and_clear_backgrounds()

sam_impl.PRE_DEFINED_ARRAY_PATH='assets/test_damaged_dd_val.npy'
sam_impl.CROP_MASK_IN_DIR=os.getcwd()+'/cars/CarDD_release/CarDD_SOD/CarDD-VAL/CarDD-VAL-Mask'
sam_impl.IN_DATA_PATH= os.getcwd()+'/cars/CarDD_release/CarDD_SOD/CarDD-VAL/CarDD-VAL-Image'
sam_impl.load_images_and_clear_backgrounds()

print('####CAR_TEST_DAMAGE_DD_SEGMENTED')