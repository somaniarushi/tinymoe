from src.train import train_loop

train_loop(
    experiment_group="noisy_topk",
    experiment_name="noisy_topk_8_9m",
    max_iters=10000,
    eval_interval=100,
    top_k=8,
    device="cuda:1",
)
