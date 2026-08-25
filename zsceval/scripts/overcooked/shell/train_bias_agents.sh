#!/bin/bash
# Train the HSP bias agents ZSC-Eval evaluates against.
#
# Each seed selects one entry from an *enumerated* w0 candidate list -- the
# product of the bracketed ranges below, filtered to at most 3 non-zero bias
# terms (train_bias_agent.py:145). For random0 that list is exactly 30 long,
# which is why the upstream seed range is 1..30: it is an exhaustive
# enumeration, not a sample. A contiguous sub-range is therefore a poor subset
# -- itertools.product varies the last dimensions fastest, so seeds 1-8 pin
# pickup_onion_from_X and pickup_onion_from_O at zero. Pass a spread seed list
# instead (e.g. "1 5 9 13 17 21 25 29"), and note that seeds beyond the
# candidate count wrap and repeat.
#
# Usage: bash shell/train_bias_agents.sh <layout> [seeds] [num_env_steps] [exp]
#   seeds           space-separated list, or "a-b" range. Default: the layout's
#                   full enumeration, as upstream.
#   num_env_steps   default 1e7 (upstream). The entropy horizons below are
#                   scaled to it, keeping the upstream ratio -- without that a
#                   short run never leaves the 0.2 exploration phase.
#   exp             default "hsp-s1", which is what extract_bias_agents_models.py
#                   looks for. Upstream sets "hsp-S1" here, and that extractor
#                   cannot find it: the W&B filter is case-sensitive.
env="Overcooked"

layout=$1
seeds_arg=$2
num_env_steps=${3:-1e7}
exp=${4:-"hsp-s1"}

entropy_coefs="0.2 0.05 0.001"
mid_ratio=0.6
if [[ "${layout}" == "small_corridor" ]]; then
    mid_ratio=0.8
fi
entropy_mid=$(awk -v s="${num_env_steps}" -v r="${mid_ratio}" 'BEGIN{printf "%.0f", s*r}')
entropy_end=$(awk -v s="${num_env_steps}" 'BEGIN{printf "%.0f", s}')
entropy_coef_horizons="0 ${entropy_mid} ${entropy_end}"
reward_shaping_horizon=$(awk -v s="${num_env_steps}" 'BEGIN{printf "%.0f", s}')

num_agents=2
algo="mappo"


if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
    # old layouts
    #! positive reward shaping for "[op]_X" may crash the training, be careful
    #! negative reward shaping for "put_X" may be meaningless
    # "put_onion_on_X",
    # "put_dish_on_X",
    # "put_soup_on_X",
    # "pickup_onion_from_X", random0_medium random0_hard
    # "pickup_onion_from_O", all_old
    # "pickup_dish_from_X",
    # "pickup_dish_from_D", all_old
    # "pickup_soup_from_X", random0 random0_medium random0_hard
    # "USEFUL_DISH_PICKUP", default
    # "SOUP_PICKUP", all_old default
    # "PLACEMENT_IN_POT", all_old default
    # "delivery", all_old
    # "STAY", all_old
    # "MOVEMENT",
    # "IDLE_MOVEMENT",
    # "IDLE_INTERACT_X",
    # "IDLE_INTERACT_EMPTY",
    # sparse_reward all_old

    w1="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"
    if [[ "${layout}" == "random0" ]]; then
        w0="0,0,0,0,[0:10],0,[0:10],[-20:0],3,5,3,0,[-0.1:0:0.1],0,0,0,0,[0.1:1]"
        seed_begin=1
        seed_max=30
    elif [[ "${layout}" == "random0_medium" ]]; then
        w0="0,0,0,[-20:0],[-20:0:10],0,[0:10],[-20:0],3,5,3,0,[-0.1:0:0.1],0,0,0,0,[0.1:1]"
        seed_begin=1
        seed_max=54
    elif [[ "${layout}" == "small_corridor" ]]; then
        w0="0,0,0,0,[-20:0:5],0,[-20:0:5],0,3,5,3,[-20:0],[-0.1:0],0,0,0,0,[0.1:1]"
        seed_begin=1
        seed_max=124
    else
        w0="0,0,0,0,[-20:0:10],0,[-20:0:10],0,3,5,3,[-20:0],[-0.1:0:0.1],0,0,0,0,[0.1:1]"
        seed_begin=1
        seed_max=176
    fi
else
    version="new"
    # 0 "put_onion_on_X",
    # 1 "put_tomato_on_X",
    # 2 "put_dish_on_X",
    # 3 "put_soup_on_X",
    # 4 "pickup_onion_from_X",
    # 5 "pickup_onion_from_O",
    # 6 "pickup_tomato_from_X",
    # 7 "pickup_tomato_from_T",
    # 8 "pickup_dish_from_X",
    # 9 "pickup_dish_from_D",
    # 10 "pickup_soup_from_X",
    # 11 "USEFUL_DISH_PICKUP",  # counted when #taken_dishes < #cooking_pots + #partially_full_pots and no dishes on the counter
    # 12 "SOUP_PICKUP",  # counted when soup in the pot is picked up (not a soup placed on the table)
    # 13 "PLACEMENT_IN_POT",  # counted when some ingredient is put into pot
    # 14 "viable_placement",
    # 15 "optimal_placement",
    # 16 "catastrophic_placement",
    # 17 "useless_placement",  # pot an ingredient to a useless recipe
    # 18 "potting_onion",
    # 19 "potting_tomato",
    # 20 "cook",
    # 21 "delivery",
    # 22 "deliver_size_two_order",
    # 23 "deliver_size_three_order",
    # 24 "deliver_useless_order",
    # 25 "STAY",
    # 26 "MOVEMENT",
    # 27 "IDLE_MOVEMENT",
    # 28 "IDLE_INTERACT",
    # 29 sparse_reward
    w1="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"
    w0="0,0,0,0,0,0,0,0,0,0,0,3,5,3,0,0,0,0,[-20:0],[-20:0],0,0,[-5:0:20],[-15:0:10],0,[-0.1:0:0.1],0,0,0,1"
    seed_begin=1
    seed_max=72
fi

# The layout blocks above set seed_begin/seed_max to the full enumeration; the
# optional argument overrides it with an explicit list or an "a-b" range.
if [[ -z "${seeds_arg}" ]]; then
    seeds=$(seq ${seed_begin} ${seed_max})
elif [[ "${seeds_arg}" =~ ^[0-9]+-[0-9]+$ ]]; then
    seeds=$(seq ${seeds_arg%-*} ${seeds_arg#*-})
else
    seeds="${seeds_arg}"
fi

# Envs per worker process, not a minibatch size. Upstream's 100 threads at
# dummy 2 is 50 worker processes, which thrashes anything smaller than a
# cluster node; throughput on this box peaks near 5.
rollout_threads=${ROLLOUT_THREADS:-12}
dummy=${DUMMY_BATCH_SIZE:-4}
eval_threads=${HSP_EVAL_THREADS:-20}

echo "layout ${layout}, exp ${exp}, steps ${num_env_steps}, entropy horizons ${entropy_coef_horizons}"
echo "seeds: $(echo ${seeds} | tr '\n' ' ')"
for seed in ${seeds};
do
    echo "seed is ${seed}:"
    python train/train_bias_agent.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}" --layout_name ${layout} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads $TRAINING_THREADS --n_rollout_threads ${rollout_threads} --dummy_batch_size ${dummy}  --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
    --overcooked_version ${version} \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --use_hsp --w0 ${w0} --w1 ${w1} --share_policy --random_index \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_recurrent_policy \
    --use_proper_time_limits \
    --save_interval 25 --log_interval 10 --use_eval --eval_interval 20 --n_eval_rollout_threads ${eval_threads} \
    --wandb_name $WANDB_ENTITY
done
