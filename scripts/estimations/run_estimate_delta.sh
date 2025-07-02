cd ./demark

# "meta-llama/Meta-Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3" 
model_id="meta-llama/Meta-Llama-3.1-8B-Instruct" 


# larger run_num/repeat_time/random_word_num give better estimation of delta
run_num=10
repeat_time=10
random_word_num=20
actual_delta=2
h=3


# gray-box setting
python estimate_delta.py \
    --model_id $model_id \
    --run_num $run_num \
    --repeat_time $repeat_time \
    --actual_delta $actual_delta \
    --random_word_num $random_word_num \
    --context_width_h $h 


# black-box setting
# python estimate_delta.py \
#     --model_id $model_id \
#     --run_num $run_num \
#     --repeat_time $repeat_time \
#     --actual_delta $actual_delta \
#     --random_word_num $random_word_num \
#     --context_width_h $h \
#     --blackbox