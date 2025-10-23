set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ppo_dataset_type="math"   # Name of the dataset
actor_template="chatml"  # chat template of actor (supported: default, chatml, llama2, llama3)
critique_template="chatml"  # chat template of critic (supported: default, chatml, llama2, llama3)
discrimination_only_step=500     # steps for stage 1(optimize discrimination)
bsz=64
freezing_actor_steps=-1

dataset="/your_constructed_dataset"
actor_model="your_actor_model_path"
critique_model="your_critique_model_path"


init_kl_coef=0.01
actor_learning_rate=5e-7
critic_learning_rate=9e-6
lr_warmup_ratio=5e-4
temperature=0.7   # Sampling temperature when RL
num_episodes=100
ppo_inner_epochs=2

save_critique_model_path="/your_output_critique_model_dir"
mkdir -p ${save_critique_model_path}


# sleep 1000
remote_rm_url="http://localhost:5000/get_reward"    # reward server url
wandb_project="your_wandb_project_name"
wandb_run_name="your_wandb_run_name"

cd RL/Critique-RL


#############################################################

#############################################################
# rm server
port=5000

python reward_with_discrimination.py \
    --critique_model ${critique_model} \
    --port ${port} \
    --dataset_type ${ppo_dataset_type} \
    --discrimination_only_step ${discrimination_only_step} \
    > reward_server${rf_mode}.log  &
sleep 50
###########################################################

###########################################################
# train critique-rl stage 1
# cd ..

read -r -d '' training_commands <<EOF
train_ppo \
   --pretrain ${critique_model} --freezing_actor_steps ${freezing_actor_steps} --advantage_estimator rloo --n_samples_per_prompt 4 \
   --ppo_dataset_type ${ppo_dataset_type} \
   --actor_model ${actor_model} \
   --actor_template ${actor_template} \
   --critique_template ${critique_template} \
   --remote_rm_url ${remote_rm_url} \
   --discrimination_only_step ${discrimination_only_step} \
   --save_path ${save_critique_model_path} \
   --save_steps 10 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 4 \
   --train_batch_size ${bsz} \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size ${bsz} \
   --max_epochs ${ppo_inner_epochs} \
   --num_episodes ${num_episodes} \
   --prompt_max_len 2048 \
   --generate_max_len 1536 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate ${actor_learning_rate} \
   --lr_warmup_ratio ${lr_warmup_ratio} \
   --critic_learning_rate ${critic_learning_rate} \
   --init_kl_coef ${init_kl_coef}\
   --prompt_data ${dataset} \
   --max_samples 100000 \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing  \
   --use_wandb True --temperature ${temperature} \
   --wandb_project ${wandb_project} \
   --wandb_run_name ${wandb_run_name}
EOF


if [[ ${1} != "slurm" ]]; then
    deepspeed --master_port 39699 --module $training_commands
fi

###########################################################