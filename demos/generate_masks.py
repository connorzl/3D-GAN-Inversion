import glob 
import os 

input_dir = '/media/data6/connorzl/pigan/S-GAN/stylegan3/pti_inversion_data/3dgan_synthetic_new/source_images'
output_dir = '/media/data6/connorzl/pigan/S-GAN/stylegan3/pti_inversion_data/3dgan_synthetic_new/source_masks'

os.makedirs(output_dir, exist_ok=True)
sources = sorted(glob.glob(os.path.join(input_dir, "*.png")))
for source in sources:
    cmd = "python generate_dataset.py"
    output_name = os.path.basename(source).split(".")[0]
    cmd += " -i " + source
    cmd += " -e " + source
    cmd += " -s " + os.path.join(output_dir, output_name)
    os.system(cmd)

