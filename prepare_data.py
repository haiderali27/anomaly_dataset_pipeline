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