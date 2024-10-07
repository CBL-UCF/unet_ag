import glob
import tqdm
import os
import shutil
import re

DS_PATH = './dataset/SCH'

DESTINATION = './dataset/'

DATASET_NAME = 'SCH'
## this script was used to format the SCH dataset to what we used for the other dataset
patientsFolder = glob.glob(DS_PATH + '/SCH*')

for patientFolder in tqdm.tqdm(patientsFolder):
    pat = re.sub('\D', '', patientFolder.split('SCH_Vol_')[-1])

    sax_img_files = glob.glob(patientFolder + '/SAX*/SA*_images.nii*')
    sax_gt_img_files = glob.glob(patientFolder + '/SAX*/SA*_segmentation*.nii')

    for img in tqdm.tqdm(sax_img_files):
        img_slice = re.sub(r'.*/(SAX\d+)_images\.nii', r'\1', img)
        destination_file = os.path.join(DESTINATION + 'training/SCH/images/', os.path.basename(f"patient{pat}_{img_slice}.nii"))
        shutil.copy2(img, destination_file)

    for gt in tqdm.tqdm(sax_gt_img_files):
        gt_slice = re.sub(r'.*/(SAX\d+)_segmentation\.nii', r'\1', gt)
        # print('#$#$#$ gtslice', gt_slice)
        destination_file = os.path.join(DESTINATION  + 'training/SCH/labels/', os.path.basename(f"patient{pat}_{gt_slice}.nii"))
        shutil.copy2(gt, destination_file)



