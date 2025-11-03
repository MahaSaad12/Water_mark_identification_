from PIL import Image
import numpy as np, torch
from torchvision import transforms
from load_modal import load_model  # <-- imports the robust loader

img_path = r"C:\Users\maha_\Job_search\pythonProject2\paired\test\images\1.png"

weights  = r"C:\Users\maha_\Job_search\pythonProject2\runs\index\train\best.pt"

tf = transforms.Compose([transforms.Resize((512,512), interpolation=Image.BILINEAR),
                         transforms.ToTensor()])

img = Image.open(img_path).convert("RGB")
x = tf(img).unsqueeze(0)

model = load_model(weights)
with torch.no_grad():
    prob = torch.sigmoid(model(x)["out"])[0,0].cpu().numpy()

print("prob stats:", float(prob.min()), float(prob.mean()), float(prob.max()))
Image.fromarray((prob*255).astype("uint8")).save("../debug_prob.png")
