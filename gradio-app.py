import gradio as gr
from mmengine import MODELS, Config
from mmengine.registry import init_default_scope

init_default_scope("mmagic")


def stable_diffusion(prompt):
    config = "configs/stable_diffusion/stable-diffusion_ddim_denoisingunet.py"
    config = Config.fromfile(config).copy()
    StableDiffuser = MODELS.build(config.model)
    StableDiffuser = StableDiffuser.to("cuda")
    image = StableDiffuser.infer(prompt)["samples"][0]

    return image


gr.Interface(
    fn=stable_diffusion, inputs="text", outputs="image", allow_flagging="never"
).launch()
