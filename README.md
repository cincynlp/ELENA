# ELENA

## Narrating Embodied Emotions via Large Vision-Language Models

This repository contains the implementation of ELENA (Embodied LVLM Emotion Narratives). This novel framework utilizes Large Vision-Language Models (LVLMs) to generate structured, multi-layered narratives of embodied emotions in images. Unlike traditional emotion recognition approaches that rely primarily on facial expressions, ELENA focuses on the physical manifestations of emotions through body language, posture, and physiological responses.

**Paper**: [Anatomy of a Feeling: Narrating Embodied Emotions via Large Vision-Language Models]()

### Key Features

1. Zero-shot structured prompting.
2. Multi-layered narratives, which are a combination of explicit and implicit descriptions, along with emotion labels, and body part identification
3. Face masking: To evaluate non-facial emotion recognition capabilities.
4. Attention visualization to analyze model focus patterns
5. Support for multiple LVLMs: Gemini 2.5 Flash, Gemma-3-12B-IT, Llama-3.2-11B-Vision-Instruct, and Llama-3.2-90B-Vision-Instruct

### Research Contributions

1. First work to systematically address embodied emotion recognition in vision-language models through structured prompting
2. Empirical evidence of persistent facial bias in LVLMs and their failure to redirect attention to informative body regions when faces are masked
3. Demonstrates significant improvements over naive prompting approaches, especially in face-masked scenarios

---

## Repository Structure

```
ELENA/
├── scripts/
│   ├── experiments/
│   └── utils/               # Shared utilities
├── attention_visualization/ # Attention map analysis tools
├── masking/          # Face masking utilities
├── responses/              # Pre-generated model outputs
│   ├── gemini_outputs/     # Gemini responses for all datasets
├── configs/                # Configuration files and prompts
├── requirements.txt        # Python dependencies
└── README.md             
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA 11.2+ (for GPU support)
- 16GB+ GPU memory (recommended for larger models)

### 1. Clone Repository

```bash
git clone https://github.com/your-username/ELENA.git
cd ELENA
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv elena_env
source elena_env/bin/activate  # On Windows: elena_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Model Setup

#### Gemini 2.5 Flash API
```bash
# Set up Google AI Studio API key
export GOOGLE_API_KEY="your_api_key_here"

# Or create a .env file
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

#### Gemma-3-12B-IT Model
```bash
# Download from Hugging Face
# Model ID: google/gemma-3-12b-it
```
You can download it here [Gemma Model HF](https://huggingface.co/google/gemma-3-12b-it)

#### Llama-3.2-11B-Vision-Instruct Model
```bash
# Download from Hugging Face Hub
# Model ID: meta-llama/Llama-3.2-11B-Vision-Instruct
```
You can download it here [Llama-11B Model](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)


### 4. YuNet Face Detection Model

Download the YuNet model for face masking:
```bash
# Download YuNet model
mkdir -p models/yunet
cd models/yunet
# YuNet model download link: [https://github.com/geaxgx/depthai_yunet]
# Place the downloaded model file here
```

---

## Dataset Setup

### Supported Datasets

1. **BESST (Bochum Emotional Stimulus Set)**
   - 1,129 controlled lab images with masked faces
   - Download: [www.rub.de/neuropsy/BESST.html]

2. **HECO (Human Emotions in Context)**
   - 9,385 natural setting images
   - Download: [https://heco2022.github.io/]

3. **EMOTIC**
   - 23,571 everyday scenario images
   - Download: [https://s3.sunai.uoc.edu/emotic/download.html]

### Dataset Preparation

```bash
# Create dataset directory structure
mkdir -p data/{besst,heco,emotic}

# Place downloaded datasets in respective folders
# data/
# ├── besst/
# │   ├── images/
# │   └── annotations/
# ├── heco/
# │   ├── images/
# │   └── annotations/
# └── emotic/
#     ├── images/
#     └── annotations/
```

---

## Usage Instructions

### Basic ELENA Experiment

#### 1. Gemini 2.5 Flash
```bash
cd scripts/gemini
python elena_gemini.py \
    --dataset_path ../../data/heco \
    --output_path ../../results/gemini_heco \
    --batch_size 20 \
    --masked_faces True
```

#### 2. Gemma-3-12B
```bash
cd scripts/gemma
python gemma_12b_exp.py \
    --img_folder ../../data/heco/images \
    --output_json_path ../../results/gemma_heco.json \
    --checkpoint_path ../../results/gemma_checkpoint.json \
    --batch_size 20
```

#### 3. Llama-3.2-11B-Vision
```bash
cd scripts/llama
python elena_llama.py \
    --dataset_path ../../data/besst \
    --model_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --output_path ../../results/llama_besst
```


### Face Masking Experiment

```bash
cd yunet_masking
python mask_faces.py \
    --input_dir ../data/heco/images \
    --output_dir ../data/heco/masked_images \
    --yunet_model ../models/yunet/yunet.onnx
```
After obtaining the masked images, you can repeat experiments 1-3 to obtain results for masked images.

### Attention Visualization

```bash
cd attention_visualization
python visualize_attention.py \
    --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
    --image_path ../data/heco/sample_image.png \
    --output_dir ../results/attention_maps
```

---

## Configuration

### Prompt Templates

ELENA uses structured prompts defined in `configs/prompts.py`:

- **System Prompt**: Defines the task and model expertise
- **User Prompt**: Specifies output format and requirements
- **Masked Prompt**: Modified version for face-masked images

### Output Format

ELENA generates structured JSON outputs with the following components:

```json
{
    "emotion": "Happiness",
    "narrative": "Contextual story incorporating body parts and emotional experience based on explicit body part and internat sensation and non-visible responses",
    "body_parts": ["shoulders", "hands", "posture"],
}
```

---

## Pre-generated Responses

The `responses/` directory contains pre-generated outputs from our experiments:

- `gemini_outputs/` - Best performing Gemini 2.5 Flash results for all datasets


---

## Troubleshooting

### Common Issues

1. **Model Download Failures**
   ```bash
   # Clear Hugging Face cache and retry
   huggingface-cli cache clear
   ```

3. **API Rate Limits (Gemini)**
   ```bash
   # Increase delay between requests. The amount of requests you can send depends on the tier you are on.
   python elena_gemini.py --api_delay 2.0
   ```



---

## Citation

If you use this work, please cite our paper:

```bibtex
@article{saim2025elena,
  title={Anatomy of a Feeling: Narrating Embodied Emotions via Large Vision-Language Models},
  author={Saim, Mohammad and Duong, Phan Anh and Luong, Cat and Bhanderi, Aniket and Jiang, Tianyu},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```



## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [saimmd@mail.uc.edu](mailto:saimmd@mail.uc.edu)
