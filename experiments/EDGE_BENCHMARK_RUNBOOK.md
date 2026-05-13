# Raspberry Pi Edge Inference Benchmark Runbook

This runbook measures deployed IDS inference on Raspberry Pi 4 and Raspberry Pi 5. The Pi runs CSV replay over precomputed flow features. It does not run LLM generation, Chroma retrieval, embeddings, or notebooks.

The benchmark records software telemetry only. CPU time, frequency, temperature, memory, and throttling are resource proxies. Do not report watts or joules per flow unless an external power meter is added.

## What This Benchmark Does

Yes, this benchmark performs inference on the datasets. It does not infer from raw packets or live traffic yet. It reads the same CSV flow-feature datasets used by the existing notebook and benchmark code, recreates the train/test split, and runs inference on the held-out test rows.

The flow is:

1. Load a dataset CSV, for example `1-cic-iot/data/sample-100000-2.csv`.
2. Recreate the same train/test split used by `experiments/11-matched-inference-benchmark.py`.
3. Train DT/RF baselines on the training rows when those conditions are selected. This happens before timing.
4. Load the LLM rule policy from `experiments/policies/llm-rule-policies.json`.
5. Run inference on the test rows:
   - `LLM_rule_policy`: applies threshold rules and majority vote.
   - `DT`: calls `DecisionTreeClassifier.predict`.
   - `RF`: calls `RandomForestClassifier.predict`.
6. Measure latency, throughput, memory, CPU time, temperature, frequency, and throttling while inference runs.
7. Write CSV/JSON results under `experiments/results/edge/`.

There are two inference modes:

- `streaming`: evaluates one dataset row at a time. This is closest to a live gateway receiving one flow at a time.
- `batch`: evaluates many rows together. This gives best-case throughput when flows can be buffered.

Use `--rows 100` only for smoke tests. When `--rows` is set, the script now selects a reproducible random sample from the held-out test split by default with `--rows-sample-seed 42`. Omit `--rows` for the real paper run so the full held-out test split is used.

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
  --rows-sample-seed 42 \
  --repeats 3 \
  --warmups 1 \
  --allow-few-repeats
```

The script writes:

```text
experiments/results/edge/pi-edge-benchmark-latest.csv
experiments/results/edge/pi-edge-benchmark-latest.json
```

For this smoke test, only 100 randomly sampled test rows are inferred. That confirms the code path works before doing a longer run.

## 5. Run Paper Benchmarks

Run each condition on each device. Keep the Pi idle except for the benchmark.

This command performs streaming inference on the full CIC-IoT2023 held-out test split. It includes all three detector conditions.

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

These commands perform batch inference on the same held-out test split, changing only the number of rows passed to each inference call.

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
