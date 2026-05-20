"""
Demo locale — Image Captioning (CLIP-L + GPT-2)
Settimana 3, Flickr30k

Avvio:
    pip install gradio
    python demo_captioning.py

Apre automaticamente http://localhost:7860
"""

import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    CLIPImageProcessor,
    GPT2Tokenizer,
)
import gradio as gr

# ── Config ────────────────────────────────────────────────────────────────────
CLIP_NAME   = "openai/clip-vit-large-patch14"
GPT2_NAME   = "gpt2"
CHECKPOINT  = Path.home() / "Documents/dev/image-cationing/checkpoints/week3_clip/best"
MAX_LENGTH  = 30
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Caricamento modello ────────────────────────────────────────────────────────
print(f"[demo] Carico il modello da {CHECKPOINT} ...")
model = VisionEncoderDecoderModel.from_pretrained(str(CHECKPOINT))
model.eval().to(DEVICE)
print(f"[demo] Modello su {DEVICE}  ✓")

tokenizer = GPT2Tokenizer.from_pretrained(GPT2_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Normalizzazione CLIP (identica al training)
clip_proc = CLIPImageProcessor.from_pretrained(CLIP_NAME)
eval_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=clip_proc.image_mean, std=clip_proc.image_std),
])

# ── Inferenza ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_caption(pil_image, num_beams: int, max_length: int, no_repeat_ngram: int):
    if pil_image is None:
        return "⚠️  Carica un'immagine prima."

    pixel_values = eval_transforms(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)

    decoder_input_ids = torch.tensor(
        [[tokenizer.bos_token_id, tokenizer.bos_token_id]], device=DEVICE
    )

    output_ids = model.generate(
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=no_repeat_ngram,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption


# ── Interfaccia Gradio ─────────────────────────────────────────────────────────
with gr.Blocks(title="Image Captioning Demo", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🖼️ Image Captioning Demo
        **Modello**: CLIP ViT-L + GPT-2 small  |  **Training**: Flickr30k  |  **BLEU-4**: 25.65 (test F8k)

        Carica qualsiasi immagine e il modello genera una didascalia in inglese.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Immagine")

            with gr.Accordion("⚙️ Parametri di generazione", open=False):
                num_beams    = gr.Slider(1, 8, value=4, step=1,  label="Beam search (più alto = più lento ma migliore)")
                max_len      = gr.Slider(10, 50, value=30, step=1, label="Lunghezza massima caption")
                no_repeat    = gr.Slider(0, 5, value=3, step=1,   label="No-repeat n-gram size")

            btn = gr.Button("✨ Genera caption", variant="primary")

        with gr.Column(scale=1):
            caption_output = gr.Textbox(
                label="Caption generata",
                lines=3,
                placeholder="La caption apparirà qui...",
            )
            gr.Markdown("---")
            gr.Markdown(
                """
                **Note**:
                - Il modello è stato addestrato su immagini in inglese → le caption sono in inglese
                - Funziona meglio su fotografie realistiche (persone, animali, paesaggi, oggetti)
                - Beam search 4 è un buon compromesso velocità/qualità
                """
            )

    btn.click(
        fn=generate_caption,
        inputs=[image_input, num_beams, max_len, no_repeat],
        outputs=caption_output,
    )

    # Submit anche su Enter nell'immagine
    image_input.upload(
        fn=generate_caption,
        inputs=[image_input, num_beams, max_len, no_repeat],
        outputs=caption_output,
    )

    gr.Markdown(
        """
        ---
        <sub>Demo locale — nessun dato viene inviato a server esterni.</sub>
        """
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,          # True se vuoi un link pubblico temporaneo
        inbrowser=True,       # apre il browser automaticamente
    )