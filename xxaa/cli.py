from functools import wraps
from pathlib import Path

import click

from .torch_profile_log import TorchProfileLog


@click.group()
def cli():
    pass


def common_output_options(f):
    @click.option("-o", "--output-file", type=click.File("w"), help="The output file.")
    @click.option(
        "--output-cpu/--no-output-cpu",
        default=True,
        help="Whether output cpu-related information.",
    )
    @click.option(
        "--output-name-length", default=-1, help="The length of the otuput name."
    )
    @click.option("--output-num-of-rows", default=-1, help="The number of output rows.")
    @click.option(
        "--output-type",
        type=click.Choice(["text", "table"]),
        default="table",
        help="The type of the output file.",
    )
    @click.option(
        "-t",
        "--output-table-type",
        type=str,
        default="excel",
        show_default=True,
        help="The format of the output table.",
    )
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@cli.command()
@click.option(
    "-i",
    "--input-log-type",
    type=click.Choice(["json", "text", "table"]),
    default="text",
    show_default=True,
    help="The type of the input log file.",
)
@click.option(
    "-1",
    "--label1",
    type=str,
    default="File1",
    show_default=True,
    help="The label of the first input log file.",
)
@click.option(
    "-2",
    "--label2",
    type=str,
    default="File2",
    show_default=True,
    help="The label of the second input log file.",
)
@common_output_options
@click.argument("file1", type=click.File("r"))
@click.argument("file2", type=click.File("r"))
def compare(
    input_log_type,
    label1,
    file1,
    label2,
    file2,
    output_file,
    output_type,
    output_table_type,
    output_name_length,
    output_num_of_rows,
    output_cpu,
):
    torch_profile_log1 = TorchProfileLog.read_from_file(file1, input_log_type)
    torch_profile_log2 = TorchProfileLog.read_from_file(file2, input_log_type)
    compared_results = torch_profile_log1.compare(
        torch_profile_log2, label1=label1, label2=label2
    )
    if output_file is None:
        suffix = "log"
        if output_type == "table":
            if output_table_type == "excel":
                suffix = "xlsx"
            elif output_table_type == "csv":
                suffix = "csv"
        default_name = f"profiling-compare-{label1}-{label2}.{suffix}"
        output_file = click.open_file(default_name, mode="w")
    compared_results.write_to_file(
        output_file=output_file,
        output_type=output_type,
        output_table_type=output_table_type,
        output_name_length=output_name_length,
        output_num_of_rows=output_num_of_rows,
        output_cpu=output_cpu,
    )


@cli.command()
@click.option(
    "-i",
    "--input-type",
    type=click.Choice(["json", "text", "table"]),
    default="text",
    show_default=True,
    help="The type of the input log file.",
)
@common_output_options
@click.argument("input_file", type=click.File("r"))
def convert(
    input_type,
    output_type,
    output_table_type,
    input_file,
    output_file,
    output_name_length,
    output_num_of_rows,
    output_cpu,
):
    torch_profile_log = TorchProfileLog.read_from_file(input_file, input_type)
    if output_file is None:
        suffix = "converted.log"
        if output_type == "table":
            if output_table_type == "excel":
                suffix = "xlsx"
            elif output_table_type == "csv":
                suffix = "csv"
        default_name = Path(input_file.name).stem + f".{suffix}"
        output_file = click.open_file(default_name, mode="w")
    torch_profile_log.write_to_file(
        output_file=output_file,
        output_type=output_type,
        output_table_type=output_table_type,
        output_name_length=output_name_length,
        output_num_of_rows=output_num_of_rows,
        output_cpu=output_cpu,
    )
