gpu_num=4

# "meta-llama/Meta-Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3" 
model_id='meta-llama/Llama-3.2-3B-Instruct' 
dipmark_alpha=0.3
delta_predicted=2
delta_reduce_multiplier=1 #eta


max_new_token=300
total_word_num=20
#mmw_mmw_story mmw_book_report dolly_cw  mmw_fake_news
prompt_set_name="dolly_cw"

save_dir="../results/watermark_removal/"

cd ./demark

save_name_suffix=$(date +%Y%m%d%H%M%S)
task_num=$gpu_num
task_idx=0
while [ $task_idx -lt $task_num ]
do
    CUDA_VISIBLE_DEVICES=$task_idx python exp_watermark_removal.py \
        --task_idx $task_idx \
        --total_task_num $task_num \
        --model_id $model_id \
        --prompt_set_name $prompt_set_name \
        --dipmark_alpha $dipmark_alpha \
        --delta_predicted $delta_predicted \
        --max_new_token $max_new_token \
        --total_word_num $total_word_num \
        --delta_reduce_multiplier $delta_reduce_multiplier \
        --save_name_suffix $save_name_suffix \
        --wm_type "dipmark" \
        --save_dir $save_dir &

    task_idx=$((task_idx + 1))
done

