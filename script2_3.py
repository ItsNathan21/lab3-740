import subprocess
import sys

FREQUENCIES = [768000, 1497600, 2112000, 3129600, 3974400, 4396800]

OUT_DIR = "task_output/"


def changeFreq(freq):
    subprocess.run(
        [sys.executable, "scripts/power_pin.py", "prime", str(freq)],
        check=True,
    )


def main(outfile):
    with open(outfile, "w") as of:
        for freq in FREQUENCIES:
            changeFreq(freq)
            print(f"\n=== Frequency: {freq} ===", file=of, flush=True)

            for workload, params in [
                ("matmul", ["-p", "size=1024"]),
                ("convolution", []),
                ("saxpy", ["-p", "size=16777216", "-p", "iterations=100"]),
            ]:
                print(f"\n--- {workload} ---", file=of, flush=True)

                subprocess.run(
                    [sys.executable, "profiler.py", workload, "cpu",
                     *params, "--core", "prime",
                     "--counters", "stalls", "--repeat", "3"],
                    stdout=of,
                    stderr=subprocess.STDOUT,
                    check=True,
                )


if __name__ == "__main__":
    main(f'{OUT_DIR}/results2_3.txt')
