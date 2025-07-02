cd ./demark

# "meta-llama/Meta-Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3" 
model_id="meta-llama/Meta-Llama-3.1-8B-Instruct" 


# larger random_word_num give better estimation of delta
random_word_num=50
actual_delta=2
actual_h=3
max_h=10
est_save_dir="../results/estimations_h"

# gray-box setting
python estimate_h.py \
    --model_id $model_id \
    --actual_h $actual_h \
    --actual_delta $actual_delta \
    --max_h $max_h \
    --random_word_num $random_word_num \
    --actual_h $actual_h \
    --est_save_dir $est_save_dir \


# black-box setting
# python estimate_h.py \
#     --model_id $model_id \
#     --actual_h $actual_h \
#     --actual_delta $actual_delta \
#     --max_h $max_h \
#     --random_word_num $random_word_num \
#     --h $h \
#     --est_save_dir $est_save_dir \
#     --blackbox




