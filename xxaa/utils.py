from itertools import accumulate
from typing import Optional, Tuple


def time_to_ms(t: str) -> float:
    # note that time in pytorch log can only be us, ms and s
    if t[-2:] == "ms":
        return float(t[:-2])
    elif t[-2:] == "us":
        return float(t[:-2]) / 1000
    else:
        return float(t[:-1]) * 1000


def get_stops_in_log_header(line: str) -> list:
    stops = [len(i) + 2 for i in line.strip().split()]
    return list(accumulate(stops))


def split_line_by_stops(line: str, stops: list) -> list:
    return [line[i:j].strip() for i, j in zip([0] + stops[:-1], stops)]
