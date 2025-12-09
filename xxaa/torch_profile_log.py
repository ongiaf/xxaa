import pandas as pd
import json
import gzip

from .constants import log_columns
from .utils import get_stops_in_log_header, split_line_by_stops, time_to_ms

# compatible with pandas 1.5.0
# https://pandas.pydata.org/docs/whatsnew/v1.5.0.html#copy-on-write
pd.options.mode.copy_on_write = True


class TorchProfileLog:
    def __init__(self, performance_table: pd.DataFrame):
        self.performance_table = performance_table

    @staticmethod
    def read_from_file(file, file_type):
        if file_type == "json":
            return TorchProfileLog.read_from_json(file)
        elif file_type == "table":
            return TorchProfileLog.read_from_table(file)
        elif file_type == "text":
            return TorchProfileLog.read_from_text(file)
        else:
            raise ValueError("Invalid file type")

    @staticmethod
    def read_from_json(file):
        data = json.load(file)

        aggregated_data = {}
        cpu_categories = ["cpu_op", "cuda_runtime", "user_annotation", "Trace", "ac2g", "null", "gpu_user_annotation"]
        cuda_categories = ["gpu_memcpy", "gpu_memset", "kernel"]
        
        # Collect raw durations in microseconds
        for event in data.get("traceEvents", []):
            # Only process 'complete' events (ph="X")
            if event.get("ph") == "X":
                name = event.get("name", "Unknown")
                cat = event.get("cat", "Unknown")
                dur = event.get("dur", 0) # Duration in microseconds

                if name not in aggregated_data:
                    aggregated_data[name] = {
                        "self_cpu_time_us": 0,
                        "self_cuda_time_us": 0,
                        "call_count": 0
                    }
                
                aggregated_data[name]["call_count"] += 1

                if cat in cpu_categories:
                    aggregated_data[name]["self_cpu_time_us"] += dur
                # Note: Some CUDA categories like "Runtime" and "Driver" also represent CPU overhead.
                # Here, we categorize based on direct GPU activity for 'Self CUDA'.
                # The assumption is that 'cpu_categories' covers CPU-side of CUDA runtime/driver calls.
                elif cat in cuda_categories:
                    aggregated_data[name]["self_cuda_time_us"] += dur

        # Convert aggregated data to a list of dicts for DataFrame
        df_rows = []
        
        total_cpu_time_us = sum(item["self_cpu_time_us"] for item in aggregated_data.values())
        total_cuda_time_us = sum(item["self_cuda_time_us"] for item in aggregated_data.values())

        for name, metrics in aggregated_data.items():
            row = {
                "Name": name,
                "# of Calls": metrics["call_count"],
                "Self CPU": metrics["self_cpu_time_us"],
                "CPU total": metrics["self_cpu_time_us"], # Placeholder: assuming self is total for now, refine if needed
                "Self CUDA": metrics["self_cuda_time_us"],
                "CUDA total": metrics["self_cuda_time_us"], # Placeholder: assuming self is total for now, refine if needed
            }
            # Calculate averages (ensure no division by zero)
            if metrics["call_count"] > 0:
                row["CPU time avg"] = row["Self CPU"] / metrics["call_count"]
                row["CUDA time avg"] = row["Self CUDA"] / metrics["call_count"]
            else:
                row["CPU time avg"] = 0.0
                row["CUDA time avg"] = 0.0

            df_rows.append(row)
        
        df = pd.DataFrame(df_rows)

        # Calculate percentages
        if total_cpu_time_us > 0:
            df["Self CPU %"] = (df["Self CPU"] / total_cpu_time_us * 100).round(2)
            df["CPU total %"] = (df["CPU total"] / total_cpu_time_us * 100).round(2)
        else:
            df["Self CPU %"] = 0.0
            df["CPU total %"] = 0.0

        if total_cuda_time_us > 0:
            df["Self CUDA %"] = (df["Self CUDA"] / total_cuda_time_us * 100).round(2)
            df["CUDA total %"] = (df["CUDA total"] / total_cuda_time_us * 100).round(2)
        else:
            df["Self CUDA %"] = 0.0
            df["CUDA total %"] = 0.0

        # Apply type conversions and time formatting similar to read_from_text
        for col in log_columns["need_to_time"]:
            if col in df.columns:
                # time_to_ms expects string like "123us", so convert numeric microseconds to this format
                df[col] = df[col].apply(lambda x: time_to_ms(f"{x}us"))
        
        for col in log_columns["need_to_int"]:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # Floating point columns like percentages are already handled by .round(2)
        # which results in float type naturally.

        # Define the desired column order based on the log file example
        expected_columns_order = [
            "Name", "Self CPU %", "Self CPU", "CPU total %", "CPU total", "CPU time avg",
            "Self CUDA", "Self CUDA %", "CUDA total", "CUDA time avg", "# of Calls"
        ]
        
        # Filter and reorder only existing columns
        final_columns = [col for col in expected_columns_order if col in df.columns]
        df = df[final_columns]
        df = df[df["Name"] != "PyTorch Profiler (0)"] # Filter out meta-event
        df = df.sort_values(by="Name").reset_index(drop=True)

        return TorchProfileLog(performance_table=df)

    @staticmethod
    def read_from_table(file):
        pass

    @staticmethod
    def read_from_text(file):
        stops = None
        headers = None
        values = []
        header_count = 0
        for line in file:
            is_header = line.strip().startswith("-") and set(line.strip()) == {"-", " "}
            if is_header:
                header_count += 1
                stops = get_stops_in_log_header(line) if stops is None else stops
                continue
            if stops is None or header_count > 2:
                continue
            if header_count == 1:
                headers = split_line_by_stops(line, stops)
            else:
                values.append(split_line_by_stops(line, stops))
        df = pd.DataFrame(values, columns=headers)

        existing_columns = [
            col for col in log_columns["need_to_int"] if col in df.columns
        ]
        df[existing_columns] = df[existing_columns].astype(int)
        df_map = lambda df, fn: df.map(fn) if hasattr(df, "map") else df.applymap(fn)
        percent_to_number = lambda x: float(x[:-1]) if x.endswith("%") else 0.0
        existing_columns = [
            col for col in log_columns["need_to_float"] if col in df.columns
        ]
        df[existing_columns] = df_map(df[existing_columns], percent_to_number)
        existing_columns = [
            col for col in log_columns["need_to_time"] if col in df.columns
        ]
        df[existing_columns] = df_map(df[existing_columns], time_to_ms)

        return TorchProfileLog(performance_table=df)

    def write_to_file(
        self,
        output_file,
        output_type="table",
        output_table_type="excel",
        output_name_length=-1,
        output_num_of_rows=-1,
        output_cpu=True,
        fill_na=True,
    ):
        df = self.performance_table
        if not output_cpu:
            df = df[[col for col in df.columns if "CPU" not in col]]
        if fill_na:
            df.fillna(0, inplace=True)
        if output_num_of_rows > 0:
            df = df.head(output_num_of_rows)
        if (
            output_type == "table"
            and output_table_type != "excel"
            and output_name_length > 0
        ):
            df["Name"] = df["Name"].map(
                lambda x: x
                if len(x) <= output_name_length
                else x[:output_name_length] + "..."
            )

        if output_type == "text":
            df.to_string(output_file, index=False)
        elif output_type == "table":
            if output_table_type == "excel":
                df.to_excel(output_file.name, index=False)
            elif output_table_type == "csv":
                df.to_csv(output_file, index=False)
            else:
                df.to_markdown(output_file, index=False, tablefmt=output_table_type)
        else:
            raise ValueError("Invalid file type")

    def compare(self, other, label1, label2):
        return TorchProfileLog(
            performance_table=pd.merge(
                self.performance_table,
                other.performance_table,
                on="Name",
                how="outer",
                suffixes=(f"_{label1}", f"_{label2}"),
            )
        )
