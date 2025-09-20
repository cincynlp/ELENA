CUDA_VISIBLE_DEVICES=0 python3 cross_attention_visualize.py \
    --model "meta-llama/Llama-3.2-11B-Vision-Instruct" \
    --image_folder "unmasked" \
    --text "What is the emotion displayed in the image?" \
    --target_token "Ġemotion" \
    --output_folder "visualize" \