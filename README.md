# CL-NeRF
Official PyTorch implementation for the paper "[CL-NeRF: Continual Learning of Neural Radiance Fields for Evolving Scene Representation Video (NeurIPS 2023)](https://wxzwxzwxz.github.io/CL-NeRF/static/pdfs/2023_NeurIPS_CLNeRF.pdf)".<br/>

## Prerequisites
- You can create the environment with:
    ```
    pip install -r requirements.txt
    ```

## Download Dataset
Dataset:
```bash
mkdir -p data
gdown --fuzzy "https://drive.google.com/file/d/1DaafLYn0qzaqWC_6avdYX1seiZm8TaD6/view?usp=sharing" -O data/kitchen_dataset.zip
unzip data/kitchen_dataset.zip -d data/
```

Expected dataset path:

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

## Train

Original NeRF:
```bash
bash experiments/original/train_kitchen_original.sh
```

Single-operation CLNeRF:
```bash
bash experiments/single_operation/ADD/train_kd_expert_mask_kitchen.sh
bash experiments/single_operation/DEL/train_kd_expert_mask_kitchen.sh
bash experiments/single_operation/MOV/train_kd_expert_mask_kitchen.sh
bash experiments/single_operation/REP/train_kd_expert_mask_kitchen.sh
```

Sequential CLNeRF:
```bash
bash experiments/sequential_operation/train_kd_expert_mask_kitchen.sh
```

## Render / Evaluate

```bash
bash experiments/inference/inference_kitchen.sh
bash experiments/inference/inference_kitchen_after_ADD_clnerf.sh
bash experiments/inference/inference_kitchen_after_DEL_clnerf.sh
bash experiments/inference/inference_kitchen_after_MOV_clnerf.sh
bash experiments/inference/inference_kitchen_after_REP_clnerf.sh
bash experiments/inference/inference_kitchen_after_SEQ_clnerf.sh
```

## Extract Mesh

```bash
bash experiments/extract_mesh.sh
```

## Notes

- Put the dataset under `./data/Kitchen/`.
- Outputs are saved under `logs/<expname>/`.


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

