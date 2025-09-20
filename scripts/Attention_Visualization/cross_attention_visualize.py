import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForVision2Seq, MllamaProcessor
import cv2
import os
import glob

def overlay_cross_attention(processor: MllamaProcessor, inputs, outputs, image: Image, layer: int, figure_path: str, target_token: str | None = None):
    """
    Overlay the average cross-attention heatmap for all layers over the original image,
    aggregating (mean) over all heads, but only over the image patches.
    Also print a dict mapping from patch index → vision_token label.
    Assumes the vision encoder produces a 14×14 grid of patches.
    """

    # where the text tokens live in the joint sequence
    tokens = processor.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    img_marker = "<|image|>"
    eos_marker = "<|eot_id|>"
    seq_start = tokens.index(img_marker) + 1
    seq_end = tokens.index(eos_marker)

    cross_attn = outputs.attentions[layer][0].to(torch.float32)   # (heads, tokens, vision_tokens)
    mean_heads = cross_attn.mean(dim=0)                           # (tgt_len, src_len)
    # only keep rows = question tokens
    qa = mean_heads[seq_start:seq_end] # last token

    if target_token is not None:
        try:
            target_token_idx = tokens[seq_start:seq_end].index(target_token)
            attention_score_for_target = qa[target_token_idx, :].cpu().numpy()
            print(f"Token '{target_token}' found at index {target_token_idx}.")
        except ValueError:
            print(f"Token '{target_token}' not found in the input sequence.")
            return
    else:
        attention_score_for_target = qa.mean(dim=0).cpu().numpy()
        print("No target token specified, using average attention scores.")
        target_token_idx = None

    H_proc = 560
    W_proc = 560
    patch_size = 14

    num_patches_h = H_proc // patch_size
    num_patches_w = W_proc // patch_size
    N_active_patches = num_patches_h * num_patches_w # e.g., 40 * 40 = 1600

    active_attention_scores = attention_score_for_target[:N_active_patches]
    attention_map_patches = active_attention_scores.reshape(num_patches_h, num_patches_w)

    # upscale the attention map to 560x560 with cv2.resize
    attention_map_patches = torch.tensor(attention_map_patches)
    attention_map_patches = cv2.resize(
        attention_map_patches.numpy(),
        (H_proc, W_proc),
        interpolation=cv2.INTER_LINEAR,
    )

    # normalize the attention map to [0, 1]
    attention_map_patches = (attention_map_patches - attention_map_patches.min()) / \
                            (attention_map_patches.max() - attention_map_patches.min())
    
    # overlay the attention map on the original image
    attention_map_patches = np.uint8(attention_map_patches * 255)
    attention_map_patches = cv2.applyColorMap(attention_map_patches, cv2.COLORMAP_JET)
    attention_map_patches = cv2.cvtColor(attention_map_patches, cv2.COLOR_BGR2RGB)
    attention_map_patches = cv2.addWeighted(
        np.array(image), 0.5, attention_map_patches, 0.5, 0
    )
    attention_map_patches = Image.fromarray(attention_map_patches)
    attention_map_patches = attention_map_patches.resize((560, 560), Image.LANCZOS)
    plt.imshow(attention_map_patches)
    plt.axis("off")
    plt.title(f"Layer {layer} - Token: {target_token if target_token else 'Average'}")
    plt.savefig(figure_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize cross-attention maps.")
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the input image folder.")
    parser.add_argument("--text", type=str, required=True, help="Input text for the model.")
    parser.add_argument("--target_token", type=str, default=None, help="Target token to visualize.")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder path for the figures.")
    args = parser.parse_args()

    layers = [3, 8, 13, 18, 23, 28, 33, 38]

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        output_attentions=True,
        return_dict=True
    )
    
    # make sure the top‐level output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # find all images in the folder
    image_paths = sorted(glob.glob(os.path.join(args.image_folder, "*.*")))

    for image_path in image_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"Processing image {image_name}...")

        # load & resize
        image = Image.open(image_path).convert("RGB").resize((560, 560))
        prompt = args.text

        # build chat‐style prompt
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=image, text=input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)

        # per‐image subfolder
        out_sub = os.path.join(args.output_folder, image_name)
        os.makedirs(out_sub, exist_ok=True)

        for layer in layers:
            print(f"  Layer {layer}...")
            figure_path = os.path.join(out_sub, f"layer_{layer}.png")
            overlay_cross_attention(processor, inputs, outputs, image, layer, figure_path, args.target_token)
            print(f"    saved → {figure_path}")

        print(f"Finished {image_name}\n")