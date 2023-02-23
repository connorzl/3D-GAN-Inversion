import glob 
import os 
from multiprocessing import Process
import time
import subprocess

input_dir = '/media/data6/connorzl/pigan/S-GAN/stylegan3/pti_inversion_data/3dgan_holland/source_images'
output_dir = '/media/data6/connorzl/pigan/S-GAN/stylegan3/pti_inversion_data/3dgan_holland/deca_source_masks'


start = 0
end = 1

os.makedirs(output_dir, exist_ok=True)
all_sources = sorted(glob.glob(os.path.join(input_dir, "*.png")))[start:end]

def process_dir(sources):
    for source in sources:
        output_name = os.path.basename(source).split(".")[0]
        p = subprocess.Popen(['python', 'generate_dataset.py', '-i', source, '-e', source, '-s', os.path.join(output_dir, output_name), '--device', 'cuda:8'])
        (output, err) = p.communicate()
        p.wait()

chunksize=1
processes = []
for job_idx in range(1):
    start = job_idx * chunksize
    end = start + chunksize
    tmp = Process(target=process_dir,  args=([all_sources[start:end]]))
    processes.append(tmp)

for p in processes:
    p.start()
    time.sleep(5)
for p in processes:
    p.join()



