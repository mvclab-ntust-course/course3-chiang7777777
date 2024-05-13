import argparse
from diffusers import AutoPipelineForText2Image
import torch
import os

parser = argparse.ArgumentParser(description="Inference for Text-to-Image Generation")
parser.add_argument("--model_dir", type=str, default="./output", help="Directory containing model weights")
parser.add_argument("--output_dir", type=str, default="./output_images", help="Directory to save output images")
parser.add_argument("--prompt", type=str, default="expansion joint", help="Text prompt for image generation")
parser.add_argument("--num", type=int, default=1, help="Number of images to generate")

args = parser.parse_args()

existing_files = len(os.listdir(args.output_dir))
pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights(args.model_dir, weight_name="pytorch_lora_weights.safetensors")

for i in range(existing_files, existing_files + args.num):
    image = pipeline(args.prompt).images[0]
    image.save(os.path.join(args.output_dir, f"{i}.png"))