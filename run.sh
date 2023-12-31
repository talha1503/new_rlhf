# Copyright 2022 The ML Fairness Gym Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

set -e
set -x

# Activate virtual environment.  If this fails, be sure that virtualenv is
# installed.
virtualenv -p python3 .
source ./bin/activate asdasda

# Install ML fairness gym requirements.
pip install -r requirements.txt

# Train baseline agent
source ./scripts/baseline.sh

# Generate trajectories
source ./scripts/sample_generator.sh

# Get Synthetic preferences
source ./scripts/generate_preferences.sh

# Train Reward Model
source ./scripts/train_reward_model.sh

# Train agent using reward model
source ./scripts/preference.sh