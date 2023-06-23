# Quick start guide for RLHF fairness

### Install the requirements
````
pip install -r requirements.txt
````

### Train baseline agent

```
sh ./scripts/baseline.sh
```

Hyperparameters:
1) For Classifier agent: <br> 
   num_steps <br>
   include_summary_stats <br>
   classifier_name <br>
   sampling_flag <br>
2) For RL agent: <br>
   rl_agent <br>
   include_summary_stats <br>
   K_epochs <br>
   lr_actor <br>
   lr_critic <br>
   max_ep_len <br>
   update_timestep <br>
   model_checkpoint_path <br>
   test_mode <br>

### Generate Trajectories
````
./scripts/sample_generator.sh
````

### Sampling summary statistics of trajectories:

For varying the values of the hyperparameters, each range follows 3 values indicating the starting value, (the ending value - step size) and the step size.
In the example below, the interest rate from 0.2, 0.4, 0.6, 0.8 and 1.0 will be sampled, same for the other hyperparameter ranges.
A total of 2 X 5 X 9 X 4 = 360 combinations of hyperparameters will be used for generating trajectories.

```
python3 experiments/lending_demo.py --num_steps 500 --sampling_flag
```

Output path: "./data/metrics/metric_trajectories_test.csv"

For changing arguments:

```
python3 experiments/lending_demo.py \
  --num_steps 500 \
  --sampling_flag \
  --policy_options "equalize_opportunity" "max_reward" \
  --interest_rate_range 0.2 1.2 0.2 \
  --bank_starting_cash_range 5000 15000 1000
  --seed_range 200 250 10
```

### Extracting preferences

#### Synthetic preferences
```

```

#### Human preferences
```

```

### Train reward model
``` 

```

### Train PPO:

```
python .\experiments\lending_demo.py --rl_agent=ppo --num_steps=500 --include_summary_stats --K_epochs=80 --lr_actor=0.0003 --lr_critic=0.001 --max_ep_len=5000 --update_timestep=200
```

### Test PPO:

```
python .\experiments\lending_demo.py --rl_agent=ppo --num_steps=500 --include_summary_stats --max_ep_len=5000 --test_mode
```
