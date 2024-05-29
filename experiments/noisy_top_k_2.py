from src.train import train_loop

train_loop(
    experiment_group="noisy_topk",
    experiment_name="noisy_topk_2_9m",
    max_iters=10000,
    eval_interval=100,
)
