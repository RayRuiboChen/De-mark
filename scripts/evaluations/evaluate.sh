openai_api_key=<>

exp_dir="./results/watermark_removal/dolly_cw/meta-llama_Llama-3.2-3B-Instruct/20250702003725"


cd ./demark


python evaluate_results.py \
    --exp_dir $exp_dir \
    --openai_api_key $openai_api_key


