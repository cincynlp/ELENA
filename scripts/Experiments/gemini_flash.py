import os
import json
import time
from datetime import datetime
import google.generativeai as genai
from PIL import Image
from collections import Counter
import glob

IMAGE_DIRECTORY = ""
OUTPUT_FILE = ""
CHECKPOINT_FILE = ""
BATCH_SIZE = 30

API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash-preview-04-17"
RATE_LIMIT_PER_MINUTE = 1000
RATE_LIMIT_PER_DAY = 10000

if not API_KEY:
    print("key environment variable not set.")
    exit()

genai.configure(api_key=API_KEY)

EMOTION_PROMPT = """
You are an expert in detecting embodied emotions in images. Embodied emotions are physical manifestations of
emotional states where:
(1) a physical movement or physiological response is evoked by an emotion, and
(2) the physical movement has no purpose other than emotion expression.

You will think step by step and analyze images:
1. Identify the visible body signals ( posture, gestures, facial expressions,).
2. Connect these signals to underlying physiological responses.
3. Determine the emotional state causing these responses.
4. Generate a backstory portrayal narrative descriptions.


For each image, think extensively to provide the following in strict JSON format:

"explicit_description": "A short explicit description focusing on visible body parts and their emotional expression",
"implicit_description": "A short implicit description capturing internal sensations (body parts which are not visible) (e.g., "My heart raced" rather than "Her eyes widened")",
"narrative": "Think of the body parts, both external and implicit ((body parts which are not visible e.g., My "heart" raced rather than My "hands" froze) involved in the emotion reaction. Based on the body parts, create a natural story involving the scene in the image. Make sure the story is natural, creative and believable. Don't announce emotions outright but allowing the reader to imply. Make up information for context. Do not be too repetitive in every sentence. Use a variety of words and phrases.",
"body_parts": "Exact same body parts mentioned in the narrative. Use the same words as in the narrative. Do not use synonyms or other words.",
"emotion": "ONLY ONE of: Happiness, Sadness, Fear, Anger, Surprise, Disgust and Neutral",

DO NOT INCLUDE ANY OTHER EXPLANATORY TEXT OUTSIDE OF JSON.
"""

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
                return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return {}
    print("No checkpoint found. Starting fresh.")
    return {}

def save_checkpoint(results, is_final=False):
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(results, f, indent=4)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=4)

        emotions = [data.get('emotion', 'Unknown') for data in results.values()]
        emotion_counts = Counter(emotions)

        print("\n--- Curr. Results Distribution ---")
        total_results = len(results)
        if total_results > 0:
            for emotion, count in emotion_counts.items():
                print(f"{emotion}: {count} ({count/total_results*100:.1f}%)")
        else:
            print("No results yet")


        if is_final:
            print(f"\nFinal results saved to {OUTPUT_FILE}")
        else:
            print(f"\nCheckpoint saved with {len(results)} images")

    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def process_image(image_path):
    filename = os.path.basename(image_path)
    # print(f"Processing {filename}...") #  print to main loop

    analysis_data = {

        "explicit_description": "N/A",
        "implicit_description": "N/A",
        "narrative": "N/A",
        "body_parts": "N/A",
        "emotion": "Unknown",
    }

    image_part = None
    raw_response = "N/A" 

    try:
        Image.open(image_path) # Just to verify the image can be opened

        image_part = genai.upload_file(path=image_path)
        content = [EMOTION_PROMPT, image_part]

        generation_config = genai.GenerationConfig(
            #temperature=TEMPERATURE,
            #top_k=TOP_K,
            # max_output_tokens=1024,
        )

        model = genai.GenerativeModel(model_name=MODEL_NAME, generation_config=generation_config)
        response = model.generate_content(content)

        if response and response.text:
            raw_response = response.text.strip()

            # clean json
            cleaned_response = raw_response
            if cleaned_response.startswith("```json\n") and cleaned_response.endswith("\n```"):
                cleaned_response = cleaned_response[len("```json\n"):-len("\n```")]

            try:
                parsed_data = json.loads(cleaned_response)

                for key in analysis_data.keys():
                    if key in parsed_data:
                        analysis_data[key] = parsed_data[key]

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for {filename}: {e}")
                print(f"  Raw response from API for {filename}:\n{raw_response}")

            except Exception as e:
                print(f"Error processing response for {filename}: {e}")
                print(f"  Raw response from API for {filename}:\n{raw_response}")

        else:
            print(f"No response text received for {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

    finally:
        try:
            if image_part:
                genai.delete_file(image_part.name)
        except Exception as e:
            print(f"Error deleting uploaded file for {filename}: {e}")

    return filename, analysis_data

def main():
    results = load_checkpoint()

    all_image_files = [
        os.path.join(IMAGE_DIRECTORY, f) for f in os.listdir(IMAGE_DIRECTORY)
        if os.path.isfile(os.path.join(IMAGE_DIRECTORY, f)) and
        f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]

    processed_filenames = set(results.keys())
    image_files_to_process = [f for f in all_image_files if os.path.basename(f) not in processed_filenames]

    print(f"Found {len(all_image_files)} total images")
    print(f"Already processed: {len(processed_filenames)}")
    print(f"Remaining to process: {len(image_files_to_process)}")

    if len(processed_filenames) >= RATE_LIMIT_PER_DAY:
        print(f"Daily limit of {RATE_LIMIT_PER_DAY} already reached. Saving current results...")
        save_checkpoint(results, is_final=True)
        return

    remaining_daily_limit = RATE_LIMIT_PER_DAY - len(processed_filenames)
    if len(image_files_to_process) > remaining_daily_limit:
        image_files_to_process = image_files_to_process[:remaining_daily_limit]

    if not image_files_to_process:
        save_checkpoint(results, is_final=True)
        return

    requests_this_minute = 0
    minute_start_time = time.time()
    batch_counter = 0
    total_processed_in_session = 0 
    total_to_process_in_session = len(image_files_to_process) 

    print(f" {total_to_process_in_session} images...")

    for img_path in image_files_to_process:
        current_time = time.time()
        time_elapsed = current_time - minute_start_time

        if requests_this_minute >= RATE_LIMIT_PER_MINUTE and time_elapsed < 60:
            wait_time = 60 - time_elapsed
            print(f"Rate limit reached ({RATE_LIMIT_PER_MINUTE}).{wait_time:.1f} seconds...")
            time.sleep(wait_time)
            minute_start_time = time.time()
            requests_this_minute = 0

        elif time_elapsed >= 60:
            minute_start_time = current_time
            requests_this_minute = 0

        filename, analysis_data = process_image(img_path)
        requests_this_minute += 1

        results[filename] = analysis_data
        total_processed_in_session += 1
        batch_counter += 1
        emotion = analysis_data.get('emotion', 'Unknown')
        print(f"Fin. {filename}: {emotion} ({total_processed_in_session}/{total_to_process_in_session} in session)")

        # Save checkpoint when batch size is reached
        if batch_counter >= BATCH_SIZE:
            print(f"\nCompleted batch of {BATCH_SIZE} images.")
            save_checkpoint(results)
            batch_counter = 0 # Reset batch counter after saving checkpoint

    if total_processed_in_session > 0:
        save_checkpoint(results, is_final=True)
        print(f" Processed {len(results)} images in total.")
    else:
        print("\nNo new images.")


if __name__ == "__main__":
    main()
