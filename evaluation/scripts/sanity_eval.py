import os
import re


def get_average_spl_and_success_rate(folder_path):
    spl_values = []
    pattern = re.compile(r"spl=([\d\.]+)")

    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            filename = filename[:-4]  # Remove .mp4 extension
            match = pattern.search(filename)
            if match:
                spl_value = float(match.group(1))
                spl_values.append(spl_value)

    if spl_values:
        average_spl = sum(spl_values) / len(spl_values)
        success_rate = sum(1 for spl in spl_values if spl > 0) / len(spl_values)
        return average_spl, success_rate
    else:
        return None, None


# Example usage:
folder_path = "/home/anjie/Projects/NaVILA/NaVILA-Stage/evaluation/eval_out/navila-llama3-8b-stage3/VLN-CE-v1/val_unseen/videos"  # Change this to your actual folder path
average_spl, success_rate = get_average_spl_and_success_rate(folder_path)
if average_spl is not None:
    print(f"Average SPL: {average_spl}")
    print(f"Success Rate: {success_rate}")
else:
    print("No SPL values found.")
