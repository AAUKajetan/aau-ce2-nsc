#!/bin/bash
# Usage: ./run-on-workers.sh "command to run on all nodes"
# Example: ./run-on-workers.sh "sudo apt install -y openjdk-17-jdk"

NODES=(10.92.0.211 10.92.0.87 10.92.0.160 10.92.0.16)
SUDO_PASS="dask"
CMD="$*"
USER="dask"
VENV_ACTIVATE="source ~/nsc_dask/dask-env/bin/activate"

if [ -z "$CMD" ]; then
  echo "Usage: $0 \"command\""
  exit 1
fi

echo "Running: $CMD"
echo "---"

for host in "${NODES[@]}"; do
  echo "=== $USER@$host ==="
  if [[ "$CMD" == sudo* ]]; then
    ssh -i ~/.ssh/id_ed25519_bdm "$USER@$host" "cd /home/$USER && $VENV_ACTIVATE && echo $SUDO_PASS | sudo -S bash -c '${CMD#sudo }'"
  elif [[ "$CMD" == nohup* ]]; then
    # Background commands - use -f to not wait
    ssh -f -i ~/.ssh/id_ed25519_bdm "$USER@$host" "cd /home/$USER && $VENV_ACTIVATE && $CMD"
  else
    ssh -i ~/.ssh/id_ed25519_bdm "$USER@$host" "cd /home/$USER && $VENV_ACTIVATE && $CMD"
  fi
done

echo "---"
echo "Done"