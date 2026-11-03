# Running experiments on the shared GCP project

This guide is for the collaborator account `nozellamila@gmail.com`. It creates a
Spot NVIDIA T4 VM in the same Frankfurt region as the project bucket, downloads
the training data, runs the paper experiment catalog, and uploads results to the
collaborator's isolated output prefix.

## Access boundaries

The collaborator can:

- create and operate Compute Engine VMs in project `ufu-test`;
- connect to VMs using OS Login;
- read the objects in the training bucket;
- attach the dedicated `nozellamila-runner` service account; and
- write objects below `runs/nozellamila/`.

The collaborator cannot administer IAM, billing, budgets, or the bucket itself,
and cannot attach the original `parkinson-trainer` service account. Compute
Engine usage is charged to `ufu-test`. Budget alerts notify the project owner but
do not impose a hard spending limit.

The Moura cohort (keyed `hemato` in code and on the bucket) contains restricted
clinical data. Do not copy it outside the authorized project or share it with
another identity.

## 1. Install and authenticate the Google Cloud CLI

Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), then:

```bash
gcloud auth login nozellamila@gmail.com
gcloud config set project ufu-test
gcloud auth list
```

The active account in the final command must be `nozellamila@gmail.com`.

Set the shared configuration used throughout this guide:

```bash
export PROJECT=ufu-test
export ZONE=europe-west3-b
export VM=nozellamila-parkinson-t4
export BUCKET=gs://ufu-test-parkinson-training-europe-west3
export RUNNER=nozellamila-runner@ufu-test.iam.gserviceaccount.com
```

## 2. Inspect the available data and previous results

```bash
gcloud storage ls "$BUCKET/data/"
gcloud storage ls "$BUCKET/runs/strict-daddb01/results/"
```

The bucket contains Zham (`kaggle`), Folador, and the authorized Moura (`hemato`)
final folders.
Existing results are read-only. New results must be written below:

```text
gs://ufu-test-parkinson-training-europe-west3/runs/nozellamila/
```

## 3. Create a Spot T4 VM

The following configuration matches the successful reproduction environment:

```bash
gcloud compute instances create "$VM" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --max-run-duration=86400s \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --service-account="$RUNNER" \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata=install-nvidia-driver=True,enable-oslogin=TRUE
```

To reproduce published `dogan_triple` cells exactly, use `--machine-type=n1-standard-8`
and check that `lscpu` reports an Intel Broadwell CPU: Dogan's results depend on the
core count and CPU model (see `REPRODUCE.md`). The other pipelines do not need this.

Spot capacity is not guaranteed. If creation reports that no T4 is available,
retry later rather than selecting a different region: keeping the VM in
`europe-west3` avoids cross-region data transfer.

Connect and verify the GPU:

```bash
gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT"
nvidia-smi
export BUCKET=gs://ufu-test-parkinson-training-europe-west3
```

## 4. Clone and install the reproduction code

Run these commands on the VM:

```bash
git clone --branch cleanup-open-source-pr1 --single-branch \
  https://github.com/NIATS-UFU/parkinson-model-training.git
cd ~/parkinson-model-training

# Optional: reproduce the exact signed run rather than the latest branch commit.
git checkout daddb01b8d59b26f971fe30bb7dbbfb8bd4f5a98

cd paper_reproduction
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e ".[train,hf]" --no-deps
```

Do not install unpinned package upgrades when reproducing the signed run.

## 5. Download the datasets and build a local manifest

Run on the VM:

```bash
mkdir -p ~/parkinson-data
gcloud storage rsync --recursive "$BUCKET/data" ~/parkinson-data

cd ~/parkinson-model-training/paper_reproduction
source .venv/bin/activate
mkdir -p results
python scripts/paper.py build-manifest \
  --data-root ~/parkinson-data \
  --out results/manifest.parquet
python scripts/paper.py reproduce --dry-run
```

Build a new manifest locally because the file paths in a manifest are specific
to the machine where it was created. The dry run should list 97 cells when all
three datasets are present.

## 6. Run, sync, and stop automatically

Create a runner on the VM. It synchronizes cached cells every two minutes,
performs a final sync even after a training failure, and shuts the VM down only
after a successful complete run.

```bash
cat > ~/run-paper-experiments.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

run_id=${1:?Usage: run-paper-experiments.sh RUN_ID}
repo=$HOME/parkinson-model-training/paper_reproduction
destination=gs://ufu-test-parkinson-training-europe-west3/runs/nozellamila/$run_id

cd "$repo"
source .venv/bin/activate

sync_results() {
  gcloud storage rsync --recursive results "$destination/results"
  gcloud storage cp "$HOME/training.log" "$destination/training.log"
}

(
  while true; do
    sync_results
    sleep 120
  done
) > "$HOME/result-sync.log" 2>&1 &
sync_pid=$!

training_status=0
python scripts/paper.py reproduce \
  --manifest results/manifest.parquet \
  --out results || training_status=$?

kill "$sync_pid" 2>/dev/null || true
wait "$sync_pid" 2>/dev/null || true
sync_results

if (( training_status != 0 )); then
  echo "Training failed with status $training_status; leaving the VM running." >&2
  exit "$training_status"
fi

python scripts/paper.py verify-catalog --results results
date -u +%FT%TZ > "$HOME/completed.txt"
gcloud storage cp "$HOME/completed.txt" "$destination/completed.txt"
sudo -n shutdown -h now
EOF

chmod +x ~/run-paper-experiments.sh
RUN_ID="$(date -u +%Y%m%d-%H%M%S)"
nohup ~/run-paper-experiments.sh "$RUN_ID" > ~/training.log 2>&1 &
echo "Results: $BUCKET/runs/nozellamila/$RUN_ID/"
```

The full catalog is idempotent: completed `cell_id` outputs are cached, so a
stopped or preempted Spot VM can resume without recomputing completed cells.
Running two GPU training cells simultaneously is not recommended on one T4; the
two processes compete for the same GPU and did not improve observed throughput.

### If the run exits non-zero

The script above leaves the VM running on a non-zero exit so you can inspect it.
Check `~/training.log` for one of these three messages before assuming a crash.

**`CUDA is required to run: …`** — the run aborted immediately, before any training.
The T4 is not visible to PyTorch. This is the guard that stops a catalog from silently
training on CPU for days. Run `nvidia-smi`: if it reports no device, the Spot VM lost
its GPU or the driver did not install, so recreate the VM (step 3). If `nvidia-smi`
works, the message will instead say the installed torch is a CPU-only build —
reinstall from `requirements-lock.txt` as in step 4. There is no override flag.

- `Huang VGG19 produced no usable gradient in any batch; the model never trained`
- `Huang VGG19 collapsed to a single-class predictor on its own training data`

Both of these are **deliberate guards**, not failures of the environment. They abort a
Huang fine-tune that finished but produced no usable model, so that a collapsed run
cannot be mistaken for a result. Re-launch the same run ID: caching means only the
failed cell is recomputed. See [`REPRODUCE.md`](REPRODUCE.md) for details.

Any other non-zero exit is a genuine error — check the log before restarting.

Monitor from the VM:

```bash
tail -f ~/training.log
watch -n 5 nvidia-smi
```

Monitor from the local computer:

```bash
gcloud compute instances describe "$VM" \
  --zone="$ZONE" --project="$PROJECT" --format='value(status)'
gcloud storage ls "$BUCKET/runs/nozellamila/"
```

## 7. Resume after Spot preemption

If the VM status becomes `TERMINATED`, restart it and launch the same run ID:

```bash
gcloud compute instances start "$VM" --zone="$ZONE" --project="$PROJECT"
gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT"
nohup ~/run-paper-experiments.sh ORIGINAL_RUN_ID > ~/training.log 2>&1 &
```

The persistent boot disk retains the repository, datasets, and cached local
results. The GCS copy provides an additional checkpoint.

## 8. Stop or delete resources manually

Stop the VM whenever it is idle:

```bash
gcloud compute instances stop "$VM" --zone="$ZONE" --project="$PROJECT"
```

Stopping ends VM and GPU charges, but the boot disk remains billable. After all
needed outputs are confirmed in GCS, delete the VM and its boot disk:

```bash
gcloud compute instances delete "$VM" \
  --zone="$ZONE" --project="$PROJECT" --delete-disks=all
```

Never delete another collaborator's VM or anything outside
`runs/nozellamila/`.
