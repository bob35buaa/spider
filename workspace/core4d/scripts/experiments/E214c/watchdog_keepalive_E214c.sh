#!/usr/bin/env bash
# E214c keepalive watchdog (session-independent backstop).
#
# User requirement: after ALL 100 rows reach cem_ok, the GPU keepalive script
# (/mnt/ali-sh-1/usr/xiayibo/.cache/run.py) MUST be hanging back on the cards.
#
# Layers: (1) the launch chain restarts keepalive when the runner exits rc=0;
# (2) THIS watchdog: polls the manifest every 60s; when cem_ok>=100 AND the
# runner is gone AND keepalive is not running (double-checked after 30s to avoid
# racing the chain), it starts the keepalive. If the runner died with <100 done
# it only logs an ALERT (resume is a human/Claude decision -- do NOT grab cards).
#
# Start:  nohup bash workspace/core4d/scripts/experiments/E214c/watchdog_keepalive_E214c.sh >/dev/null 2>&1 &
# Stop:   kill <pid>  (recorded in the log header)
set -u
cd /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider
MAN=workspace/core4d/results/E214c/manifests/e214c_partwise_cem.tsv
LOG=logs/E214c/watchdog.log
KEEPALIVE=/mnt/ali-sh-1/usr/xiayibo/.cache/run.py

echo "[watchdog] started $(date -Iseconds) pid=$$" >> "$LOG"
alerted=0
while true; do
  ok=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)h[$i]=i;next} $h["status"]=="cem_ok"{n++}END{print n+0}' "$MAN" 2>/dev/null)
  # Anchored patterns: only real processes, NOT the bash -c chain wrapper whose
  # cmdline merely CONTAINS these strings.
  ka=$(ps -eo args= 2>/dev/null | grep -cE "^[^ ]*python[^ ]* /mnt/ali-sh-1/usr/xiayibo/\.cache/run\.py")
  runner=$(ps -eo args= 2>/dev/null | grep -cE "^[^ ]*python[^ ]* .*run_ablation_cem\.py")
  mjwp=$(ps -eo args= 2>/dev/null | grep -cE "^[^ ]*python[^ ]* -u examples/run_mjwp\.py")

  if [ "${ok:-0}" -ge 100 ] && [ "$runner" -eq 0 ] && [ "$ka" -eq 0 ]; then
    sleep 30  # double-check: don't race the launch chain's own restart
    ka=$(ps -eo args= 2>/dev/null | grep -cE "^[^ ]*python[^ ]* /mnt/ali-sh-1/usr/xiayibo/\.cache/run\.py")
    if [ "$ka" -eq 0 ]; then
      echo "[watchdog] 100/100 cem_ok, runner gone, keepalive absent -> starting keepalive $(date -Iseconds)" >> "$LOG"
      nohup python "$KEEPALIVE" > /dev/null 2>&1 &
      echo "[watchdog] keepalive started pid=$!" >> "$LOG"
    fi
  fi

  if [ "$runner" -eq 0 ] && [ "$mjwp" -eq 0 ] && [ "${ok:-0}" -lt 100 ]; then
    if [ "$alerted" -eq 0 ]; then
      echo "[watchdog] ALERT $(date -Iseconds): runner dead with cem_ok=$ok/100 -- needs resume (rerun run_ablation_cem.py); keepalive intentionally NOT started" >> "$LOG"
      alerted=1
    fi
  else
    alerted=0
  fi

  echo "[watchdog] $(date -Iseconds) cem_ok=${ok:-?}/100 runner=$runner mjwp=$mjwp keepalive=$ka" >> "$LOG"
  sleep 60
done
