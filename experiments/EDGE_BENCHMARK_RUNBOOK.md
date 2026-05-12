# Raspberry Pi Edge Inference Benchmark Runbook

This runbook measures deployed IDS inference on Raspberry Pi 4 and Raspberry Pi 5. The Pi runs CSV replay over precomputed flow features. It does not run LLM generation, Chroma retrieval, embeddings, or notebooks.

The benchmark records software telemetry only. CPU time, frequency, temperature, memory, and throttling are resource proxies. Do not report watts or joules per flow unless an external power meter is added.

## 1. Prepare Each Raspberry Pi

Use Raspberry Pi OS 64-bit Lite on both devices where possible.

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip libraspberrypi-bin
```

Set the CPU governor to `performance` before paper runs.

```bash
for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance | sudo tee "$governor"
done
```

Confirm the setting and initial throttling state.

```bash
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort -u
vcgencmd get_throttled
vcgencmd measure_temp
```

Use active cooling if possible. If `vcgencmd get_throttled` is not `throttled=0x0`, record that run as thermally constrained or discard it for the main comparison.

## 2. Copy the Repo and Data

From the host machine:

```bash
rsync -av --exclude '.git' --exclude '.venv' ./ pi@<pi-host>:~/iot-llm-hids-pg/
```

Copy dataset CSVs that are not already in the repo. CIC-IoT2023 can start with:

```bash
rsync -av 1-cic-iot/data/sample-100000-2.csv \
  pi@<pi-host>:~/iot-llm-hids-pg/1-cic-iot/data/
```

For all-dataset runs, also stage the WUSTL-IIoT, TON-IoT, Bot-IoT, and UNSW-NB15 CSVs in the same paths expected by `experiments/11-matched-inference-benchmark.py`.

## 3. Create the Python Environment

On the Pi:

```bash
cd ~/iot-llm-hids-pg
python3 -m venv .venv-edge
. .venv-edge/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy pandas scikit-learn psutil
```

The edge benchmark only needs these minimal runtime packages.

## 4. Run Smoke Tests

Run unit tests first.

```bash
python3 -m unittest tests/test_edge_inference.py
```

Run a 100-row benchmark smoke test.

```bash
python3 experiments/13-edge-inference-benchmark.py \
  --dataset cic \
  --condition llm \
  --mode all \
  --rows 100 \
  --repeats 3 \
  --warmups 1 \
  --allow-few-repeats
```

The script writes:

```text
experiments/results/edge/pi-edge-benchmark-latest.csv
experiments/results/edge/pi-edge-benchmark-latest.json
```

## 5. Run Paper Benchmarks

Run each condition on each device. Keep the Pi idle except for the benchmark.

```bash
python3 experiments/13-edge-inference-benchmark.py \
  --dataset cic \
  --condition all \
  --mode streaming \
  --repeats 30 \
  --warmups 5 \
  --telemetry-hz 2 \
  --cooling-note "active fan"
```

Run batch mode at the planned batch sizes.

```bash
for size in 16 128 1024 full; do
  python3 experiments/13-edge-inference-benchmark.py \
    --dataset cic \
    --condition all \
    --mode batch \
    --batch-size "$size" \
    --repeats 30 \
    --warmups 5 \
    --telemetry-hz 2 \
    --cooling-note "active fan"
done
```

When the remaining datasets are staged:

```bash
python3 experiments/13-edge-inference-benchmark.py \
  --dataset all \
  --condition all \
  --mode all \
  --batch-size full \
  --repeats 30 \
  --warmups 5 \
  --skip-missing \
  --cooling-note "active fan"
```

## 6. Collect Results

From the host machine:

```bash
rsync -av pi@<pi-host>:~/iot-llm-hids-pg/experiments/results/edge/ \
  experiments/results/edge/<pi-model-name>/
```

Keep separate folders for Pi 4 and Pi 5. Record board revision, RAM size, OS image, cooling, governor, and whether any throttling occurred.

## 7. Paper Reporting

Use `pi-edge-benchmark-*.csv` for tables. Recommended columns:

- device and CPU model
- condition
- mode and batch size
- median, IQR, p95, and p99 milliseconds per flow
- throughput in flows per second
- peak RSS
- CPU time per 1k flows
- max CPU temperature
- median ARM frequency
- throttling status

Report software telemetry as a resource proxy. Only add power or energy-per-flow claims after measuring input power with a USB-C inline meter, INA219/INA260, or comparable external instrument.
