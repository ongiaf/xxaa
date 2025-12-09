import torch

def test_func():
    input1 = torch.ones(1024, 1024, dtype=torch.float32).to(device="cuda:0")
    input2 = torch.ones(1024, 1024, dtype=torch.float32).to(device="cuda:0")
    print(input1.device, input2.device)
    return torch.matmul(input1, input2).cpu()


def trace_handler(prof):
    name = "tests/logs/Profiling-A100"
    prof.export_chrome_trace(name + ".json.gz")
    prof_table = prof.key_averages().table(
        sort_by="self_cuda_time_total",
        max_name_column_width=10000, max_src_column_width=10000, row_limit=-1
    )
    with open(name + ".log", "w") as f:
        f.write(prof_table)


def main():
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
        on_trace_ready=trace_handler,
    )
    test_func()
    with prof:
        for _ in range(5):
            test_func()
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
