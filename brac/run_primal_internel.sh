# Copyright 2019 The Google Research Authors.
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
TRAIN_ALPHA=True
ALPHA=0.1
PLR=3e-04
VALUE_PENALTY=True
DIVERGENCE=kl
ENV=HalfCheetah-v2
DATA=pure_two_policies1
DATA_SUB_DIR=1
B_CKPT=/data1/yrli/tmp/offlinerl/learn/$ENV/$DATA/n1000000/bc/0/0/agent_behavior
python train_offline.py \
  --alsologtostderr --sub_dir=auto \
  --env_name=$ENV \
  --agent_name=braci_primal \
  --data_name=$DATA \
  --data_sub_dir=$DATA_SUB_DIR \
  --total_train_steps=500000 \
  --gin_bindings="braci_primal_agent.Agent.behavior_ckpt_file='$B_CKPT'" \
  --gin_bindings="braci_primal_agent.Agent.alpha=$ALPHA" \
  --gin_bindings="braci_primal_agent.Agent.train_alpha=$TRAIN_ALPHA" \
  --gin_bindings="braci_primal_agent.Agent.value_penalty=$VALUE_PENALTY" \
  --gin_bindings="braci_primal_agent.Agent.n_div_samples=10" \
  --gin_bindings="braci_primal_agent.Agent.divergence_name='$DIVERGENCE'" \
  --gin_bindings="train_eval_offline.model_params=(((300, 300), (200, 200),), 2)" \
  --gin_bindings="train_eval_offline.batch_size=256" \
  --gin_bindings="train_eval_offline.optimizers=(('adam', 1e-3), ('adam', $PLR), ('adam', 1e-3))" \
