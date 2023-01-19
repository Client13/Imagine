import os
os.system("pip install --upgrade git+https://github.com/huggingface/diffusers.git transformers accelerate scipy xformers gradio -q") 
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
os.system("pip install flask")
from flask import Flask, send_file
#from flask_ngrok import run_with_ngrok
app = Flask(__name__)
run_with_ngrok(app)   

from flask import request
generator = torch.Generator("cuda").manual_seed(1024)

@app.route('/')
def hello_world():
    prompt=request.args.get('q')
    
    image = pipe(prompt,num_inference_steps=100, guidance_scale = 10,generator=generator).images[0]    
    image.save("sd_image.png")

    return send_file("sd_image.png", mimetype='image/gif')   #return res

def txt2img(prompt):

  image = pipe(prompt, height=768, width=768, guidance_scale = 10).images[0]
    
  image.save("sd_image.png")

  return
if __name__ == '__main__':
    app.run(debug=True)
