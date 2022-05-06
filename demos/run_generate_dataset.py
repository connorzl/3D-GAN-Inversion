import glob 
import os 

all_images = sorted(glob.glob("/media/data6/connorzl/pigan/S-GAN/stylegan3/pti_inversion_data/3dgan_synthetic_new/source_images/*"))
all_videos = sorted(glob.glob("/media/data6/connorzl/pigan/S-GAN/stylegan3/pti_inversion_data/3dgan_targets/vox_frames_padded/*"))

#all_images = sorted(glob.glob("/orion/u/connorzl/projects/DECA/Dataset_large/3dgan_teaser_images/*"))
#all_videos = sorted(glob.glob("/orion/u/connorzl/projects/DECA/Dataset_large/obama/*"))

output_dir = "/media/data6/connorzl/pigan/S-GAN/stylegan3/pti_inversion_data/3dgan_synthetic_new/deca_source_images"

#assert(len(all_images) == 500)
#assert(len(all_videos) == 500)

#start = 0
#end = 100

#start = 100
#end = 200

#start = 200
#end = 300

#start = 300 
#end = 400

#start = 400
#end = 500

start = 0
end = 2

for i in range(start, end):
    cmd = "python generate_dataset.py"
    output_name = os.path.basename(all_images[i]).split(".")[0] + "_" + os.path.basename(all_videos[i])
    cmd += " -i " + all_images[i]
    cmd += " -e " + all_videos[i]
    cmd += " -s " + os.path.join(output_dir, output_name)
    os.system(cmd)

