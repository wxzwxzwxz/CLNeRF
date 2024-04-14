# CL-NeRF
Official PyTorch implementation for the paper "[CL-NeRF: Continual Learning of Neural Radiance Fields for Evolving Scene Representation Video (NeurIPS 2023)](https://arxiv.org/pdf/2309.04814.pdf)".<br/>

## Prerequisites
- You can create an environment with:
    ```
    pip install -r requirements.txt
    ```

## Download Pre-trained Weights
 Download the pre-trained models from the drive and unzip them into the project's root directory for later testing. Refer to the directory structure example provided: Kitchen contains the pre-trained NeRF model, and Kitchen_ADD_clnerf is fine-tuned with new images following the ADD operation.:
```
├── logs 
│   ├── Kitchen
│   ├── Kitchen_ADD_clnerf
│   ├── ...
```

## Download Dataset
 Download the datasets from the drive and unzip them into the project's root directory for training. Refer to the provided example for the directory structure:
```
├── data 
│   ├── Kitchen
│   │   ├── original
│   │   ├── sequential_operation
│   │   ├── single_operation
│   │   │   ├── ADD
│   │   │   ├── DEL
│   │   │   ├── MOV
│   │   │   ├── REP
│   ├── Whiteroom 
│   ├── ...
```

## Test
We provide an example using the ADD operation in the Kitchen dataset. Start by downloading the pre-trained weights and dataset. Then, execute the following script: 
```
bash ./experiments/inference/inference_kitchen_after_ADD_clnerf.sh
```
When finished, results are saved to `logs/Kitchen_ADD_clnerf/renderonly_test_stage1_newtask_209999` and `logs/Kitchen_ADD_clnerf/renderonly_test_stage1_oldtask_209999`. To use different operations or datasets, replace 'ADD' and 'kitchen' accordingly.

## Train
First download the dataset. Then,
```
bash ./experiments/single_operation/ADD/train_kd_expert_mask_kitchen.sh
```

## Citation
If you utilize the code, datasets, or concepts from our paper in your research, please kindly cite:
```
@article{wu2024cl,
  title={CL-NeRF: Continual Learning of Neural Radiance Fields for Evolving Scene Representation},
  author={Wu, Xiuzhe and Dai, Peng and Deng, Weipeng and Chen, Handi and Wu, Yang and Cao, Yan-Pei and Shan, Ying and Qi, Xiaojuan},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
