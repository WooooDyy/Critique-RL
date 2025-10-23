base_model_path="/your_base_actor_or_critique_model_path"
template="LLaMA-Factory_template_name"
model_output_dir="/your_output_model_dir"
epoch=5

cd LLaMA-Factory
sleep 3

echo "SFT started."
MASTER_PORT=39101 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path ${base_model_path} \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template ${template} \
    --flash_attn fa2 \
    --dataset_dir "/your_LLaMA-Factory_data_directory_path" \
    --dataset "your_dataset_name_set_in_LLaMA-Factory_dataset_json" \
    --cutoff_len 4096 \
    --learning_rate 5e-06 \
    --num_train_epochs ${epoch} \
    --max_samples 400000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 5000 \
    --warmup_steps 0 \
    --optim adamw_torch \
    --packing False \
    --report_to none \
    --report_to none \
    --output_dir /${model_output_dir}\
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --deepspeed "your_deepspeed_config_file" \
    --save_only_model \
    --overwrite_output_dir