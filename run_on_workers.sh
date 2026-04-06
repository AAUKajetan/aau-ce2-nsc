#!/bin/bash
# Usage: ./run-on-workers.sh [-P] "command to run on all nodes"
# Example: ./run-on-workers.sh "sudo apt install -y openjdk-17-jdk"
# Use -P flag for parallel execution across all nodes

NODES=(10.92.0.211 10.92.0.87 10.92.0.160 10.92.0.16)
SUDO_PASS="dask"
USER="dask"
VENV_ACTIVATE="source ~/nsc_dask/dask-env/bin/activate"
PARALLEL=false

# Parse -P flag
if [[ "$1" == "-P" ]]; then
  PARALLEL=true
  shift
fi

CMD="$*"

if [ -z "$CMD" ]; then
  echo "Usage: $0 [-P] \"command\""
  echo "  -P  Run command on all nodes in parallel"
  exit 1
fi

echo "Running: $CMD"
[[ "$PARALLEL" == true ]] && echo "(parallel mode)"
echo "---"

run_on_host() {
  local host="$1"
  echo "=== $USER@$host ==="
  if [[ "$CMD" == sudo* ]]; then
    ssh -i ~/.ssh/id_ed25519_bdm "$USER@$host" "cd /home/$USER && $VENV_ACTIVATE && echo $SUDO_PASS | sudo -S bash -c '${CMD#sudo }'"
  elif [[ "$CMD" == nohup* ]]; then
    ssh -f -i ~/.ssh/id_ed25519_bdm "$USER@$host" "cd /home/$USER && $VENV_ACTIVATE && $CMD"
  else
    ssh -i ~/.ssh/id_ed25519_bdm "$USER@$host" "cd /home/$USER && $VENV_ACTIVATE && $CMD"
  fi
}

for host in "${NODES[@]}"; do
  if [[ "$PARALLEL" == true ]]; then
    run_on_host "$host" &
  else
    run_on_host "$host"
  fi
done

[[ "$PARALLEL" == true ]] && wait

echo "---"
echo "Done"