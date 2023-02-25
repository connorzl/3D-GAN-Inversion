# Modified Implementation of DECA: Detailed Expression Capture and Animation (SIGGRAPH2021)

This is a modified repository of the [official Pytorch implementation of DECA](https://github.com/yfeng95/DECA), used by [3D GAN Inversion for Controllable Portrait Image Animation](https://arxiv.org/abs/2203.13441). 

DECA reconstructs a 3D head model with detailed facial geometry from a single input image. The resulting 3D head model can be easily animated. Please refer to the [arXiv paper](https://arxiv.org/abs/2012.04012) for more details.

## Getting Started
Clone the repo:
  ```bash
  git clone https://github.com/connorzl/DECA
  cd DECA
  ```  

### Requirements
* DECA environment  
  ```
  conda env create --file=environment.yml
  ```
* pytorch3d
  ```
  git clone https://github.com/facebookresearch/pytorch3d.git
  cd pytorch3d && pip install -e . 
  ```

### Usage
1. Prepare data   
    a. download [FLAME model](https://flame.is.tue.mpg.de/download.php), choose **FLAME 2020** and unzip it, copy 'generic_model.pkl' into ./data  
    b. download [DECA trained model](https://drive.google.com/file/d/1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje/view?usp=sharing), and put it in ./data (**no unzip required**)  
    c. (Optional) follow the instructions for the [Albedo model](https://github.com/TimoBolkart/BFM_to_FLAME) to get 'FLAME_albedo_from_BFM.npz', put it into ./data

2. Preprocess data for 3D GAN Inversion:
    ```
    python generate_dataset.py -i source_images  -e target_images -s output --device cuda:0 
    ```   
    to visualize the predicted depth, face mask, face mask rendered source image with target expression, and fully rendered source image with target expression (see output folder in this repository for example outputs).   
 
3. Run 3D GAN Inversion:
  ```
  python run_pti.py --experiment_name exp_dir --input_data_path output  --gpu 0 --input_pose_path source_cameras.json --logging_root logs
  ```

## Citation
If you find this work useful to your research, please consider citing:
```
@inproceedings{DECA:Siggraph2021,
  title={Learning an Animatable Detailed {3D} Face Model from In-The-Wild Images},
  author={Feng, Yao and Feng, Haiwen and Black, Michael J. and Bolkart, Timo},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH)}, 
  volume = {40}, 
  number = {8}, 
  year = {2021}, 
  url = {https://doi.org/10.1145/3450626.3459936} 
}

@article{lin20223d,
  title={3d gan inversion for controllable portrait image animation},
  author={Lin, Connor Z and Lindell, David B and Chan, Eric R and Wetzstein, Gordon},
  journal={arXiv preprint arXiv:2203.13441},
  year={2022}
}
```
## License
This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE](https://github.com/YadiraF/DECA/blob/master/LICENSE) file.
By downloading and using the code and model you agree to the terms in the [LICENSE](https://github.com/YadiraF/DECA/blob/master/LICENSE). 
