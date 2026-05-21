## Download
Dataset:
```bash
mkdir -p data
gdown --fuzzy "https://drive.google.com/file/d/1DaafLYn0qzaqWC_6avdYX1seiZm8TaD6/view?usp=sharing" -O data/kitchen_dataset.zip
unzip data/kitchen_dataset.zip -d data/
```

Expected dataset path:
```text
./data/Kitchen/
```

## Install

```bash
conda create -n clnerf python=3.8 -y
conda activate clnerf

# Install PyTorch for your CUDA version first.
# Example:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install gdown PyMCubes lpips
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

