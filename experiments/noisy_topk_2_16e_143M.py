from src.train import train_loop

train_loop(
    experiment_group="noisy_topk",
    experiment_name="noisy_topk_2_16experts_134M",
    max_iters=10000,
    eval_interval=100,
    top_k=2,
    n_experts=16,
    block_size=512,
    batch_size=16,
    n_embed=256,
    n_head=16,
    n_layer=16,
    device="cuda:3",
)
