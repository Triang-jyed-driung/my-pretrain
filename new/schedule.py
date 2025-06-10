import math

# 自定义调度函数
def cosine_lr_lambda(current_step, total_steps, warmup_steps, lr, lr_end):
    ratio = lr_end / lr
    one_minus_ratio = (lr - lr_end) / lr
    if current_step < warmup_steps and warmup_steps > 0:
        return ratio + one_minus_ratio * current_step / warmup_steps
    # progress <= 1
    progress = min(1, (current_step - warmup_steps) / max(1, total_steps - warmup_steps))
    return ratio + one_minus_ratio * 0.5 * (1.0 + math.cos(math.pi * progress))
