config_dict = {
    "batch_size": 128,
    "epochs": 30,
    "pretrain_epochs": 30,
    "heuristic_lr": 0.1,
    "heuristic_epochs": 20,
    "heuristic_threshold": 1e-2,
    "itr_lr": 0.1,
    "initial_prune_rate": 0.9,
    "one_time_prune_rate": 0.8,
    "itr_epochs": 30,
    "ft_lr": 1e-3,
    "pr_lr": 1e-2,
    "pretrain_lr": 1e-3,
    "pretrained_model_file": "pretrained_model.pt",
    "ft_label": 0,
    "pr_label": 1,
    "bias_key": 'bias',
    "weight_key": 'weight',
    "activate_key": 'activate_flag'
}