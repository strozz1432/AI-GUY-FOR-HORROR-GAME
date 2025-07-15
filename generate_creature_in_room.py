import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import os

print("cwd:", os.getcwd())
print("files:", os.listdir())

# load img
ROOM_IMAGE_PATH = r'C:\Users\RAZER\Desktop\AIGUY\room.jpg'  # img in
OUTPUT_IMAGE_PATH = 'room_with_creature.png'

if not os.path.exists(ROOM_IMAGE_PATH):
    raise FileNotFoundError(f"Input image '{ROOM_IMAGE_PATH}' not found.")

room_img = cv2.imread(ROOM_IMAGE_PATH)
room_img_rgb = cv2.cvtColor(room_img, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(room_img_rgb)
img_np = np.array(img_pil)

# get d
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
midas.to(device)

midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = midas_transforms.small_transform

transformed = transform(img_np)
if isinstance(transformed, dict):
    input_batch = transformed['image'].to(device)
else:
    input_batch = transformed.to(device)
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_pil.size[::-1],
        mode='bicubic',
        align_corners=False
    ).squeeze()
    depth_map = prediction.cpu().numpy()

# norm d
min_depth = np.min(depth_map)
max_depth = np.max(depth_map)
depth_map_norm = (255 * (depth_map - min_depth) / (max_depth - min_depth)).astype(np.uint8)

# make guy
creature_prompt = "cartoon monster, full body, on transparent background, floor shadow, cute, digital art"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe = pipe.to(device)

creature_image = pipe(creature_prompt, guidance_scale=7.5).images[0]
creature_image = creature_image.convert('RGBA')
creature_image.save('creature_debug.png')

# make sure creature fits horizontally
if creature_image.width >= depth_map_norm.shape[1]:
    scale = (depth_map_norm.shape[1] - 1) / creature_image.width
    new_w = int(creature_image.width * scale)
    new_h = int(creature_image.height * scale)
    creature_image = creature_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

creature_arr = np.array(creature_image)
print('Creature alpha min/max:', np.min(creature_arr[..., 3]), np.max(creature_arr[..., 3]))
creature_mask = creature_arr[..., 3] > 128

# stick guy bg
floor_band = depth_map_norm[depth_map_norm.shape[0] * 2 // 3:]
floor_depth = np.percentile(floor_band, 90)

floor_y = depth_map_norm.shape[0] - creature_image.height // 2 - 10
floor_x = np.random.randint(0, depth_map_norm.shape[1] - creature_image.width)

room_img_rgba = np.dstack([room_img_rgb, np.full(room_img_rgb.shape[:2], 255, dtype=np.uint8)])

for y in range(creature_image.height):
    for x in range(creature_image.width):
        ry = floor_y + y
        rx = floor_x + x
        if ry >= room_img_rgba.shape[0] or rx >= room_img_rgba.shape[1]:
            continue
        if creature_mask[y, x]:
            if depth_map_norm[ry, rx] >= floor_depth:
                room_img_rgba[ry, rx, :3] = creature_arr[y, x, :3]
                room_img_rgba[ry, rx, 3] = 255

# output
result_img = Image.fromarray(room_img_rgba)
result_img.save(OUTPUT_IMAGE_PATH)
print(f"Saved output to {OUTPUT_IMAGE_PATH}")

plt.imshow(result_img)
plt.axis('off')
plt.show() 