import torch
from diffusers.loaders import AttnProcsLayers
from transformers import CLIPTextModel, CLIPTokenizer
from modules.beats.BEATs import BEATs, BEATsConfig
from modules.AudioToken.embedder import FGAEmbedder
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers import StableDiffusionPipeline
import numpy as np
import gradio as gr


class AudioTokenWrapper(torch.nn.Module):
    """Simple wrapper module for Stable Diffusion that holds all the models together"""

    def __init__(
        self,
        lora,
        device,
    ):

        super().__init__()
        # Load scheduler and models
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", revision=None
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet", revision=None
        )
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae", revision=None
        )

        checkpoint = torch.load(
            'models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')
        cfg = BEATsConfig(checkpoint['cfg'])
        self.aud_encoder = BEATs(cfg)
        self.aud_encoder.load_state_dict(checkpoint['model'])
        self.aud_encoder.predictor = None
        input_size = 768 * 3
        self.embedder = FGAEmbedder(input_size=input_size, output_size=768)

        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        self.aud_encoder.eval()

        if lora:
            # Set correct lora layers
            lora_attn_procs = {}
            for name in self.unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith(
                    "attn1.processor") else self.unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet.config.block_out_channels[block_id]

                lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size,
                                                          cross_attention_dim=cross_attention_dim)

            self.unet.set_attn_processor(lora_attn_procs)
            self.lora_layers = AttnProcsLayers(self.unet.attn_processors)
            self.lora_layers.eval()
            lora_layers_learned_embeds = 'models/lora_layers_learned_embeds.bin'
            self.lora_layers.load_state_dict(torch.load(lora_layers_learned_embeds, map_location=device))
            self.unet.load_attn_procs(lora_layers_learned_embeds)

        self.embedder.eval()
        embedder_learned_embeds = 'models/embedder_learned_embeds.bin'
        self.embedder.load_state_dict(torch.load(embedder_learned_embeds, map_location=device))

        self.placeholder_token = '<*>'
        num_added_tokens = self.tokenizer.add_tokens(self.placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {self.placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
        self.placeholder_token_id = self.tokenizer.convert_tokens_to_ids(self.placeholder_token)
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))


def greet(audio):
    audio = audio[-1].astype(np.float32, order='C') / 32768.0
    weight_dtype = torch.float32
    prompt = 'a photo of <*>'

    audio_values = torch.unsqueeze(torch.tensor(audio), dim=0).to(device).to(dtype=weight_dtype)
    if audio_values.ndim == 1:
        audio_values = torch.unsqueeze(audio_values, dim=0)
    aud_features = model.aud_encoder.extract_features(audio_values)[1]
    audio_token = model.embedder(aud_features)

    token_embeds = model.text_encoder.get_input_embeddings().weight.data
    token_embeds[model.placeholder_token_id] = audio_token.clone()

    pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        tokenizer=model.tokenizer,
        text_encoder=model.text_encoder,
        vae=model.vae,
        unet=model.unet,
    ).to(device)
    image = pipeline(prompt, num_inference_steps=40, guidance_scale=7.5).images[0]
    return image


if __name__ == "__main__":

    lora = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AudioTokenWrapper(lora, device)

    description = """<p>
    This is a demo of <a href='https://pages.cs.huji.ac.il/adiyoss-lab/AudioToken' target='_blank'>AudioToken: Adaptation of Text-Conditioned Diffusion Models for Audio-to-Image Generation</a>.<br><br>
    A novel method utilizing latent diffusion models trained for text-to-image-generation to generate images conditioned on audio recordings. Using a pre-trained audio encoding model, the proposed method encodes audio into a new token, which can be considered as an adaptation layer between the audio and text representations.<br><br>
    For more information, please see the original <a href='https://arxiv.org/abs/2305.13050' target='_blank'>paper</a> and <a href='https://github.com/guyyariv/AudioToken' target='_blank'>repo</a>.
    </p>"""

    examples = [
        # ["assets/train.wav"],
        ["assets/dog barking.wav"],
        ["assets/airplane taking off.wav"],
        # ["assets/electric guitar.wav"],
        # ["assets/female sings.wav"],
    ]

    demo = gr.Interface(
        fn=greet,
        inputs="audio",
        outputs="image",
        title='AudioToken',
        description=description,
        # examples=examples
    )
    demo.launch()
