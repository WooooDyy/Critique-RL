dataset_name="openai/gsm8k"   # Path of the dataset
dataset_type="gsm8k"    # Name of the dataset
sample_num=1   # Number of sampling iterations
reserved_new_data=1   # Number of new samples to retain
response_temperature=0   # Sampling temperature when inferencing
critique_temperature=0   # Sampling temperature when critiquing
refinement_temperature=0 # Sampling temperature when refining
ITER_NUM=0   # Number of iterations
MV_NUM=1
ONLY_FINAL_SEQUENTIAL=0
EXP_NAME="gsm8k_stage2"   # Experiment identifier
TEST_USE_CRITIC=1   # Whether to use a critic (0 = No, 1 = Yes)
TEST_KNOW_ANSWER=0  # Whether the critic knows answer (0 = No, 1 = Yes)
# Note: to evaluate performance without critic, set TEST_USE_CRITIC=0, TEST_KNOW_ANSWER=1
actor_model_name="/your_actor_model_name_after_sft"

critique_template="chatml" # chat template of critic (supported: default, chatml, llama2, llama3)
actor_template="chatml" # chat template of actor (supported: default, chatml, llama2, llama3)
critic_model_names=(
    "/your_critique_model_path"
)

cd RL/Critique-RL
for critic_model_name in ${critic_model_names[@]}; do
    results_file="${critic_model_name}/result_file_${dataset_type}_${EXP_NAME}_test_${sample_num}-${MV_NUM}_${TEST_USE_CRITIC}${TEST_KNOW_ANSWER}.json"
    echo ${results_file}
    
    echo "Evaluating start."
    sleep 5
    python inference.py \
        --actor_name ${actor_model_name} \
        --dataset_name ${dataset_name} \
        --dataset_type ${dataset_type} \
        --temperature ${response_temperature} \
        --sample_num ${sample_num} \
        --results_file ${results_file} \
        --mode 'test' \
        --test_know_answer ${TEST_KNOW_ANSWER} \
        --need_false_data 0 \
        --template ${actor_template}
    wait
    for ((MV = 1; MV <= MV_NUM; MV++))
    do
        echo "MV ${MV} started"
        if [ "${TEST_USE_CRITIC}" -eq 1 ]; then
            sleep 5
            python critic.py \
                    --critic_name ${critic_model_name} \
                    --temperature ${critique_temperature} \
                    --sample_num 1 \
                    --template ${critique_template}
            wait 
            sleep 5

            python inference.py \
                    --actor_name ${actor_model_name} \
                    --dataset_name ${dataset_name} \
                    --dataset_type ${dataset_type} \
                    --temperature ${refinement_temperature} \
                    --sample_num 1 \
                    --results_file ${results_file} \
                    --mode 'new' \
                    --reserved_new_data ${reserved_new_data} \
                    --need_false_data 0 \
                    --template ${actor_template}
            wait
            
            sleep 5
            python test_filter.py \
                --mode 'sequential'
        fi
    done
    
    if [ "${ONLY_FINAL_SEQUENTIAL}" -eq 1 ]; then
        python test_filter.py \
            --results_file ${results_file} \
            --mode 'only_final_sequential'
    fi

    echo "Evaluating Episode ${episode} finished."
    sleep 5
    echo ${results_file}
    echo ${critic_model_name}
    echo "sample_num ${sample_num}"

    python test_filter.py \
        --dataset_name ${dataset_name} \
        --dataset_type ${dataset_type} \
        --results_file ${results_file} \
        --mode "passk"
    wait 
done
wait