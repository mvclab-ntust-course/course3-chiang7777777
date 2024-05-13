# 使用說明

內容主要參考自 [Hugging Face Diffusers 文檔](https://huggingface.co/docs/diffusers/en/training/lora)。
以下為個人使用流程

## Install the library
```bash    
git clone https://github.com/mvclab-ntust-course/course3-chiang7777777.git
cd diffusers
pip install .
```

## Navigate to the example folder & install dependencies
```bash    
cd examples/text_to_image
pip install -r requirements.txt
```

## Initialize Accelerate environment
```bash    
accelerate config
```
In "Do you wish to use FP16 or BF16 (mixed precision)?"
choose "BF16"

## Training 
### Put training data under ``` ./data ```
My training be like:

<img src="https://hackmd.io/_uploads/ryTUd-ymC.png" alt="image" style="width: 50%; height: auto;">

### Login huggingface & wandb
```bash    
# if you have not installed
pip install huggingface-cli
pip install wandb

huggingface-cli login --token $HUGGINGFACE_TOKEN
wandb login
```

### Launch the script
```bash    
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./output"
export HUB_MODEL_ID="expansion_joint-lora"
export TRAIN_DATA_DIR="./data"

accelerate launch --mixed_precision="bf16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=1e-03 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="expansion joint." \
  --seed=1337 
```
### Training loss
![image](https://hackmd.io/_uploads/S1qq5ZymC.png)

## Inference
```bash    
python inference.py
```
Relevant parameters :
* --model_dir : Directory containing model weights, default="./output"
* --output_dir : Directory to save output images, default="./output_images"
* --prompt : Text prompt for image generation, default="expansion joint"
* --num : Number of images to generate, default=1






