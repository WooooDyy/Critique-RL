inference_dataset_path="lighteval/math"  # Path of dataset
inference_dataset_type="math"   # Name of the dataset
inference_results_file="/your_path_to_save_inference_result"
inference_sample_num=10   # Number of sampling iterations when inferencing
inference_temperature=0.7 # Sampling temperature when inferencing
actor_template="chatml"  # chat template of actor (supported: default, chatml, llama2, llama3)
ppo_prompt_path="/your_directory_to_save_inference_data(false_temp, all_inference)"

# inference_results_file: contains correct inference result 
# ppo_prompt_path/false_temp.json: contains incorrect inference result
# ppo_prompt_path/all_inference.json: inference result including both correct and incorrect

actor_model="/your_actor_model_path_after_sft"

cd RL/Critique-RL
python inference.py --actor_name ${actor_model} \
        --dataset_name ${inference_dataset_path} \
        --dataset_type ${inference_dataset_type} \
        --temperature ${inference_temperature} \
        --sample_num ${inference_sample_num} \
        --results_file ${inference_results_file}  --need_false_data 1 --ppo_prompt_path ${ppo_prompt_path}\
        --mode 'inference' \
        --template ${actor_template} > inference.log &
wait