import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import pandas as pd
import os
import json
import re
from pathlib import Path

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"processed_files": [], "results": {}}

def save_checkpoint(checkpoint_path, processed_files, results):
    checkpoint = {
        "processed_files": processed_files,
        "results": results
    }
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=4, ensure_ascii=False)

def extract_json_from_response(response):
    try:
        json_match = re.search(r'(\{[\s\S]*\})', response)
        if json_match:
            json_str = json_match.group(1)
            result = json.loads(json_str)
            return result
        else:
            return None
    except json.JSONDecodeError:
        print(f"Failed to parse JSON  {response}")
        return None

def process_folder(f_path, model, processor, system_prompt, user_prompt, checkpoint_path, output_json_path, batch_size=100):
    checkpoint = load_checkpoint(checkpoint_path)
    results = checkpoint["results"]
    processed_files = set(checkpoint["processed_files"])
    
    img_ext = ['.jpg', '.jpeg', '.png']
    img_files = [
        f for f in Path(f_path).iterdir()
        if f.suffix.lower() in img_ext
    ]
    
    total_images = len(img_files)
    processed_count = len(processed_files)
    
    print(f"Total images to process: {total_images}")
    print(f"Already processed: {processed_count}")
    
    # Filter out already processed files
    img_files = [f for f in img_files if f.name not in processed_files]
    
    batch_counter = 0
    for idx, img_path in enumerate(img_files):
        try:
            image = Image.open(img_path)
            img_filename = img_path.name
                        
            messages = [
                [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ],
            ]
            
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)
           
            prompt_len = inputs.input_ids.shape[-1]
            output = model.generate(**inputs, max_new_tokens=500)
            output_tokens = output[0, prompt_len:].tolist()
            decoded_output = processor.decode(output_tokens, skip_special_tokens=True)
            
            json_result = extract_json_from_response(decoded_output)
            
            if json_result:
                results[img_filename] = json_result
                processed_files.add(img_filename)
                
                # Print just the emotion for tracking yk
                emotion = json_result.get("emotion", "Unknown")
                print(f"[{processed_count + 1}/{total_images}] Processed {img_filename}: Emotion={emotion}")
            else:
                results[img_filename] = {"error": "Failed to extract JSON"}
                processed_files.add(img_filename)
            
            processed_count += 1
            batch_counter += 1
            
            if batch_counter >= batch_size:
                print(f"Saving checkpoint after processing {batch_size} images...")
                save_checkpoint(checkpoint_path, list(processed_files), results)
                
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                
                print(f"Checkpoint saved. Total processed are: {processed_count}/{total_images}")
                batch_counter = 0
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            results[img_path.name] = {"error": str(e)}
            processed_files.add(img_path.name)
            processed_count += 1
            batch_counter += 1
    
    save_checkpoint(checkpoint_path, list(processed_files), results)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    return results

system_prompt = """
You are an expert in detecting embodied emotions in images. Embodied emotions are physical manifestations of
emotional states where:
(1) a physical movement or physiological response is evoked by an emotion, and
(2) the physical movement has no purpose other than emotion expression.

You will analyze images and respond ONLY in valid JSON format with no additional text.
"""

user_prompt = """
Analyze this image and provide the following in strict JSON format:

{
    "explicit_description": "A short explicit description focusing on visible body parts and their emotional expression",
    "implicit_description": "A short implicit description capturing internal sensations (body parts which are not visible) (e.g., "My heart raced" rather than "Her eyes widened")",
    "narrative": "A very concise one line story NECESSARILY CONTAINING THE BODY PART influencing the emotion. It should include the scene and context of the image. Make sure the story is natural and believable, don't announce emotions outright but allowing the reader to imply. Make up information if needed.",
    "body_parts": "The specific clear body parts (explicit) involved in expressing the emotion",
    "emotion": "ONLY ONE of: Happiness, Sadness, Fear, Anger, Surprise, Disgust and Neutral",
    "valence": 1-10 indicating the positivity/negativity of the emotion,
    "arousal": 1-10 indicating the intensity of the emotion,
    "dominance": 1-10 indicating the level of control/power
}

ONLY RESPOND WITH VALID JSON. DO NOT INCLUDE ANY OTHER TEXT OUTSIDE OF JSON.
"""

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

img_folder = ""
output_json_path = ""
checkpoint_path = ""
batch_size = 20

results = process_folder(
    img_folder,
    model,
    processor,
    system_prompt,
    user_prompt,
    checkpoint_path,
    output_json_path,
    batch_size
)

count = len(results)
emotion_counts = {}
for filename, data in results.items():
    if isinstance(data, dict) and "emotion" in data:
        emotion = data["emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
