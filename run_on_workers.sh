#!/bin/bash
# Usage: ./run-on-workers.sh "command to run on all nodes"
# Example: ./run-on-workers.sh "sudo apt install -y openjdk-17-jdk"

NODES=(10.92.0.211 10.92.0.87 10.92.0.160 10.92.0.16)
CMD="$*"

if [ -z "$CMD" ]; then
  echo "Usage: $0 \"command\""
  exit 1
fi

echo "Running: $CMD"
echo "---"

for host in "${NODES[@]}"; do
  echo "=== $host ==="
  if [[ "$CMD" == sudo* ]]; then
    ssh "ubuntu@$host" "cd /home/hadoop && sudo bash -c '${CMD#sudo }'"
  else
    ssh "ubuntu@$host" "cd /home/hadoop && $CMD"
  fi
done

echo "---"
echo "Done"