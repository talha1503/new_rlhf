# Training the reward model
python .\scripts\train_reward_model.py \
    --preference_data_path=../data/ \
    --model_eval_output_path=./data/ \
    --reward_model_name=100 \
    --save_dir=./models/ \
    --training_curve_path=./models/

# Fine tuning RL agent