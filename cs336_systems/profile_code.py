from cs336_basics.transformer import TransformerLM
from cs336_basics.training_loop import ModelParams
from cs336_basics.training_utils import cross_entropy, AdamW

import torch
import json
import pandas as pd
import statistics
from tqdm import trange
from timeit import default_timer as dt
from argparse import ArgumentParser

BATCH_SIZE = 4
CONTEXT_LENGTH = 512
VOCAB_SIZE = 10_000

torch.manual_seed(666)

def run_profile_timeit(
    model: TransformerLM,
    batch: torch.Tensor,
    warmup_steps: int,
    steps: int,
    regime: int,
    optimizer: torch.optim.Optimizer | None = None,
    target_batch: torch.Tensor | None = None
) -> dict[str, list[float]]:

    for _ in range(warmup_steps):
        model(batch)

    forward_times = []
    if regime >= 1:
        backward_times = []
    if regime == 2:
        optimizer_times = []

    for step in trange(steps):
        time_before = dt()
        print(f"before {step} forward: ", torch.cuda.memory_allocated())
        logits = model(batch)
        forward_times.append(dt() - time_before)
        print(f"after {step} forward: ", torch.cuda.memory_allocated())
        
        if regime >= 1:
            loss = cross_entropy(logits, target_batch)
            time_before = dt()
            print(f"before {step} backward: ", torch.cuda.memory_allocated())
            if regime == 2:
                optimizer.zero_grad()
            loss.backward()
            backward_times.append(dt() - time_before)
            print(f"after {step} backward: ", torch.cuda.memory_allocated())

        if regime == 2:
            time_before = dt()
            print(f"before {step} op_step: ", torch.cuda.memory_allocated())
            optimizer.step()
            optimizer_times.append(dt() - time_before)
            print(f"after {step} op_step: ", torch.cuda.memory_allocated())
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    result = {
        "forward": forward_times
    }
    if regime >= 1:
        result["backward"] = backward_times
    if regime == 2:
        result["optimizer"] = optimizer_times

    return result


def generate_data(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.randint(high=VOCAB_SIZE, size=(BATCH_SIZE, CONTEXT_LENGTH), device=device)
    target = torch.randint(high=VOCAB_SIZE, size=(BATCH_SIZE, CONTEXT_LENGTH), device=device)
    return data, target


def bench_all(path_to_configs: str, w_steps: int, n_steps: int, regime: int, output_path: str):
    configs = []
    with open(path_to_configs) as f:
        for line in f:
            if line.strip(): 
                config = json.loads(line)
                configs.append(config)

    config_results = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
    # configs = configs[:1]
    for config in configs:
        # model_config = ModelParams(**config)
        model = TransformerLM(**config, device=device)
        batch, target = generate_data(device)

        additional_params = {"target_batch": target}
        if regime == 2:
            optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.99), eps=1e-6, weight_decay=0.1)
            additional_params["optimizer"] = optimizer

        config_result = run_profile_timeit(
            model, 
            batch, 
            warmup_steps=w_steps, 
            steps=n_steps, 
            regime=regime, 
            **additional_params
        )
        config_results.append(config_result)

    df_rows = []
    for config, config_res in zip(configs, config_results):
        df_row = config
        df_row["batch_size"] = BATCH_SIZE
        df_row["warmup_steps"] = w_steps
        df_row["n_steps"] = n_steps

        df_row["forward_mean_ms"] = statistics.mean(config_res["forward"]) * 1000
        df_row["forward_std_ms"] = statistics.stdev(config_res["forward"]) * 1000

        if regime >= 1:
            df_row["backward_mean_ms"] = statistics.mean(config_res["backward"]) * 1000
            df_row["backward_std_ms"] = statistics.stdev(config_res["backward"]) * 1000

            fb_times = [f_t + b_t for f_t, b_t in zip(config_res["forward"], config_res["backward"])]
            df_row["forward-backward_mean_ms"] = statistics.mean(fb_times) * 1000
            df_row["forward-backward_std_ms"] = statistics.stdev(fb_times) * 1000

        if regime == 2:
            df_row["optimizer_mean_ms"] = statistics.mean(config_res["optimizer"]) * 1000
            df_row["optimizer_std_ms"] = statistics.stdev(config_res["optimizer"]) * 1000

            fbo_times = [f_t + b_t + o_t for f_t, b_t, o_t in zip(config_res["forward"], config_res["backward"], config_res["optimizer"])]
            df_row["forward-backward-optimizer_mean_ms"] = statistics.mean(fbo_times) * 1000
            df_row["forward-backward-optimizer_std_ms"] = statistics.stdev(fbo_times) * 1000
        
        df_rows.append(df_row)

    df = pd.DataFrame(df_rows)
    df.to_csv(output_path + ".csv", index=False)
    df.to_latex(output_path + ".tex", index=False)
    
    print(df)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--input", type=str, required=True)
    argparser.add_argument("--w_steps", type=int, default=5, required=False)
    argparser.add_argument("--n_steps", type=int, default=10, required=False)
    argparser.add_argument("--regime", type=int, choices=[0, 1, 2], default=0, required=True)
    argparser.add_argument("--output", type=str, required=True)

    parser = argparser.parse_args()
    bench_all(parser.input, parser.w_steps, parser.n_steps, parser.regime, parser.output)
