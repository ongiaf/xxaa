log_columns = dict()

log_columns["cpu_related"] = [
    "Self CPU %",
    "Self CPU",
    "CPU total %",
    "CPU total",
    "CPU time avg",
]
log_columns["need_to_float"] = ["Self CPU %", "CPU total %", "Self CUDA %"]
log_columns["need_to_int"] = ["# of Calls"]
log_columns["need_to_time"] = [
    "Self CPU",
    "CPU total",
    "CPU time avg",
    "Self CUDA",
    "CUDA total",
    "CUDA time avg",
]
