from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch
import os


cache_dir = os.path.join(os.path.dirname(__file__), "sd_cache")
os.makedirs(cache_dir, exist_ok=True)


pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    cache_dir=cache_dir
)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
init_image = Image.open('C:\\Users\\RAZER\\Desktop\\AIGUY\\room.jpg').convert("RGB")


desc = "satan, full body, standing in the corner of the room, realistic, dramatic lighting, blends with background"
negative = "do not change background, do not change room, do not change furniture, only add figure, no distortion, no blur, no extra objects, no extra people, no artifacts, no painting style"


gen = pipe(
    prompt=desc,
    image=init_image,
    strength=0.6,
    guidance_scale=7.5,
    negative_prompt=negative
)
result = gen.images[0]
result.save("room_with_creature_img2img.png")
print("saved: room_with_creature_img2img.png") 