python ./scripts/train_reward_model.py \
 --preference_data_path="./data/synthetic_decisions.csv" \
 --model_eval_path="./data/reward_model_eval.json" \
 --num_epochs=2 \
 --reward_model_name='reward_model' \
 --save_dir='./models/' \
 --training_curve_path="./data/reward_model_loss.png" \
 --n_sample_points=50