layouts=$1
for layout in ${layouts};
do
    echo ${layout}
    mkdir -p ${POLICY_POOL}/${layout}/policy_config
    cp ${PYTHONPATH}/results/Overcooked/${layout}/mappo/store_config_mlp/run1/policy_config.pkl ${POLICY_POOL}/${layout}/policy_config/mlp_policy_config.pkl
    cp ${PYTHONPATH}/results/Overcooked/${layout}/rmappo/store_config_rnn/run1/policy_config.pkl ${POLICY_POOL}/${layout}/policy_config/rnn_policy_config.pkl
done
