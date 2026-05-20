#!/usr/bin/env bash
# Register one or more SLURM job IDs for email notifications on status change.
# Usage:   jobs/scripts/track_job.sh <jobid> [jobid ...]
#          jobs/scripts/track_job.sh --list           # show active watchers
#          jobs/scripts/track_job.sh --stop <jobid>   # stop a watcher
#
# Recipient address is read from <repo>/.env (key SLURM_WATCH_EMAIL).
# Override via env: SLURM_WATCH_EMAIL=... jobs/scripts/track_job.sh ...
#
# Other env overrides:
#   SLURM_WATCH_INTERVAL  (default: 30 seconds)
#   SLURM_WATCH_STATE_DIR (default: <repo>/.slurm-watch)
#   NO_COLOR=1            disable ANSI colors
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
STATE_DIR="${SLURM_WATCH_STATE_DIR:-$REPO_ROOT/.slurm-watch}"
INTERVAL="${SLURM_WATCH_INTERVAL:-30}"
mkdir -p "$STATE_DIR"

# Pull just SLURM_WATCH_EMAIL out of .env — don't source the file blindly.
load_env_email() {
    local envfile="$REPO_ROOT/.env" val
    [[ -f "$envfile" ]] || return 0
    val=$(grep -E '^[[:space:]]*SLURM_WATCH_EMAIL[[:space:]]*=' "$envfile" \
          | tail -n1 \
          | sed -E 's/^[^=]*=[[:space:]]*//; s/^"(.*)"$/\1/; s/^'\''(.*)'\''$/\1/' \
          | sed -E 's/[[:space:]]+#.*$//')
    [[ -n "${SLURM_WATCH_EMAIL:-}" ]] && return 0
    [[ -n "$val" ]] && export SLURM_WATCH_EMAIL="$val"
}

require_email() {
    load_env_email
    if [[ -z "${SLURM_WATCH_EMAIL:-}" ]]; then
        echo "error: SLURM_WATCH_EMAIL not set. Add it to $REPO_ROOT/.env (e.g. SLURM_WATCH_EMAIL=you@example.com) or export it." >&2
        exit 1
    fi
}

# ---------- colors (TTY only) -------------------------------------------------
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    C_RESET=$'\033[0m'; C_DIM=$'\033[2m'; C_BOLD=$'\033[1m'
    C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YLW=$'\033[33m'
    C_BLU=$'\033[34m'; C_CYN=$'\033[36m'; C_MAG=$'\033[35m'
else
    C_RESET=''; C_DIM=''; C_BOLD=''
    C_RED=''; C_GRN=''; C_YLW=''; C_BLU=''; C_CYN=''; C_MAG=''
fi

state_color() {
    case "$1" in
        PENDING|CONFIGURING)         printf '%s' "$C_YLW" ;;
        RUNNING)                     printf '%s' "$C_GRN" ;;
        COMPLETED)                   printf '%s%s' "$C_BOLD" "$C_GRN" ;;
        FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE)
                                     printf '%s' "$C_RED" ;;
        GONE|COMPLETING)             printf '%s' "$C_DIM" ;;
        *)                           printf '%s' "$C_CYN" ;;
    esac
}

state_glyph() {
    case "$1" in
        PENDING|CONFIGURING)         echo "[PEND]" ;;
        RUNNING)                     echo "[RUN ]" ;;
        COMPLETED)                   echo "[DONE]" ;;
        FAILED|NODE_FAIL|BOOT_FAIL|OUT_OF_MEMORY)
                                     echo "[FAIL]" ;;
        TIMEOUT|DEADLINE)            echo "[TIME]" ;;
        CANCELLED)                   echo "[CANC]" ;;
        COMPLETING)                  echo "[ ...]" ;;
        GONE)                        echo "[GONE]" ;;
        *)                           echo "[ ?  ]" ;;
    esac
}

# ---------- scontrol parsing --------------------------------------------------
get_field() {
    local key="$1" out="$2"
    awk -v k="$key" '
        BEGIN { RS="[ \n]+"; FS="=" }
        $1 == k { sub("^" k "=", ""); print; exit }
    ' <<<"$out"
}

human_default() {
    local v="$1" fallback="${2:--}"
    [[ -z "$v" || "$v" == "(null)" || "$v" == "Unknown" ]] && echo "$fallback" || echo "$v"
}

# ---------- list / stop -------------------------------------------------------
list_watchers() {
    shopt -s nullglob
    local rows=() job pid state
    for pidfile in "$STATE_DIR"/*.pid; do
        job=$(basename "$pidfile" .pid)
        pid=$(cat "$pidfile" 2>/dev/null || echo "")
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$pidfile"
            continue
        fi
        local info name
        info=$(squeue -h -j "$job" -o "%T|%j" 2>/dev/null || true)
        if [[ -n "$info" ]]; then
            state="${info%%|*}"
            name="${info#*|}"
        else
            state="GONE"
            # Fall back to scontrol so we still get a name for completed jobs.
            name=$(scontrol show job "$job" 2>/dev/null \
                | awk 'BEGIN{RS="[ \n]+"; FS="="} $1=="JobName"{print $2; exit}')
        fi
        [[ -z "$name" ]] && name="-"
        rows+=("$job|$state|$name|$pid|$STATE_DIR/$job.log")
    done
    if [[ ${#rows[@]} -eq 0 ]]; then
        printf '%s(no active watchers)%s\n' "$C_DIM" "$C_RESET"
        return
    fi
    printf '%sACTIVE WATCHERS%s\n' "$C_BOLD" "$C_RESET"
    printf '  %-10s  %-10s  %-20s  %-8s  %s\n' "JOB" "STATE" "NAME" "PID" "LOG"
    printf '  %s%s%s\n' "$C_DIM" "$(printf '%.0s-' {1..96})" "$C_RESET"
    local row log col name
    for row in "${rows[@]}"; do
        IFS='|' read -r job state name pid log <<<"$row"
        col=$(state_color "$state")
        printf '  %-10s  %s%-10s%s  %-20s  %-8s  %s%s%s\n' \
            "$job" "$col" "$state" "$C_RESET" "$name" "$pid" "$C_DIM" "$log" "$C_RESET"
    done
}

stop_watcher() {
    local job="$1"
    local pidfile="$STATE_DIR/$job.pid"
    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile")
        kill "$pid" 2>/dev/null || true
        rm -f "$pidfile"
        printf '  %s[-]%s stopped  job %s%s%s  pid %s\n' \
            "$C_RED" "$C_RESET" "$C_BOLD" "$job" "$C_RESET" "$pid"
    else
        printf '  %s[!]%s no watcher registered for job %s\n' \
            "$C_YLW" "$C_RESET" "$job"
    fi
}

# ---------- mail send (no recipient in ps args) -------------------------------
send_mail() {
    local to="$1" subj="$2" body="$3"
    local sender
    sender="${USER:-$(whoami)}@$(hostname -f 2>/dev/null || hostname)"
    {
        printf 'From: %s\n' "$sender"
        printf 'To: %s\n' "$to"
        printf 'Subject: %s\n' "$subj"
        printf 'Content-Type: text/plain; charset=UTF-8\n'
        printf '\n'
        printf '%s\n' "$body"
    } | /usr/sbin/sendmail -t
}

# ---------- watcher loop (runs detached as `--watch <jobid>`) -----------------
watch_job() {
    local job="$1"
    require_email
    local prev="" cur info
    local sep="==============================================================="
    while true; do
        cur=$(squeue -h -j "$job" -o "%T" 2>/dev/null || true)
        [[ -z "$cur" ]] && cur="GONE"

        if [[ "$cur" != "$prev" ]]; then
            info=$(scontrol show job "$job" 2>/dev/null || true)
            local name part nodes acct user submit start end runtime tlimit reason stdout stderr workdir
            name=$(human_default    "$(get_field JobName    "$info")")
            part=$(human_default    "$(get_field Partition  "$info")")
            nodes=$(human_default   "$(get_field NodeList   "$info")" "-")
            acct=$(human_default    "$(get_field Account    "$info")")
            user=$(human_default    "$(get_field UserId     "$info")")
            submit=$(human_default  "$(get_field SubmitTime "$info")")
            start=$(human_default   "$(get_field StartTime  "$info")" "pending")
            end=$(human_default     "$(get_field EndTime    "$info")" "pending")
            runtime=$(human_default "$(get_field RunTime    "$info")" "0")
            tlimit=$(human_default  "$(get_field TimeLimit  "$info")")
            reason=$(human_default  "$(get_field Reason     "$info")" "None")
            stdout=$(human_default  "$(get_field StdOut     "$info")")
            stderr=$(human_default  "$(get_field StdErr     "$info")")
            workdir=$(human_default "$(get_field WorkDir    "$info")")

            local prev_disp="${prev:-NEW}"
            local glyph subj body
            glyph=$(state_glyph "$cur")
            subj=$(printf '%s SLURM %s  %s -> %s  (%s)' \
                "$glyph" "$job" "$prev_disp" "$cur" "$name")
            body=$(cat <<EOF
SLURM job status change
$sep

  Job ID       $job
  Name         $name
  Transition   $prev_disp  ->  $cur
  Partition    $part
  Node(s)      $nodes
  Account      $acct
  User         $user
  Reason       $reason

  Submitted    $submit
  Started      $start
  Ends         $end
  Runtime      $runtime  /  $tlimit

  Workdir      $workdir
  stdout       $stdout
  stderr       $stderr

$sep
  Host  $(hostname -f 2>/dev/null || hostname)
  Time  $(date -Iseconds)
EOF
)
            echo "$(date -Iseconds)  $job  ${prev_disp} -> ${cur}  ${name}" >> "$STATE_DIR/$job.log"
            if ! send_mail "$SLURM_WATCH_EMAIL" "$subj" "$body"; then
                echo "$(date -Iseconds)  mail failed for $job" >> "$STATE_DIR/$job.log"
            fi
            prev="$cur"
        fi
        [[ "$cur" == "GONE" ]] && break
        sleep "$INTERVAL"
    done
    rm -f "$STATE_DIR/$job.pid"
}

# ---------- entrypoint --------------------------------------------------------
usage() {
    cat <<EOF
${C_BOLD}track_job.sh${C_RESET}  -  email me when SLURM jobs change status

  ${C_CYN}$(basename "$0")${C_RESET} <jobid> [jobid ...]   register watcher(s)
  ${C_CYN}$(basename "$0")${C_RESET} --list                show active watchers
  ${C_CYN}$(basename "$0")${C_RESET} --stop <jobid>        cancel a watcher

  recipient: \$SLURM_WATCH_EMAIL  (loaded from <repo>/.env)
  poll every ${INTERVAL}s
EOF
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

case "$1" in
    --list|-l)    list_watchers; exit 0 ;;
    --stop)       shift; stop_watcher "${1:?need jobid}"; exit 0 ;;
    -h|--help)    usage; exit 0 ;;
    --watch)      shift; watch_job "${1:?need jobid}"; exit 0 ;;
esac

require_email
printf '%sregistering watchers%s  (poll %ss, recipient via .env)\n' \
    "$C_BOLD" "$C_RESET" "$INTERVAL"

for job in "$@"; do
    if [[ ! "$job" =~ ^[0-9]+$ ]]; then
        printf '  %s[!]%s skip: %s is not a numeric job id\n' \
            "$C_YLW" "$C_RESET" "$job"
        continue
    fi
    pidfile="$STATE_DIR/$job.pid"
    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        printf '  %s[=]%s already tracking job %s%s%s  pid %s\n' \
            "$C_DIM" "$C_RESET" "$C_BOLD" "$job" "$C_RESET" "$(cat "$pidfile")"
        continue
    fi
    # Self-call in --watch mode. No secrets in argv; child loads .env itself.
    setsid nohup bash "$SCRIPT_DIR/track_job.sh" --watch "$job" \
        </dev/null >>"$STATE_DIR/$job.log" 2>&1 &
    pid=$!
    echo "$pid" > "$pidfile"
    printf '  %s[+]%s tracking job %s%s%s  pid %s  %slog %s%s\n' \
        "$C_GRN" "$C_RESET" "$C_BOLD" "$job" "$C_RESET" "$pid" \
        "$C_DIM" "$STATE_DIR/$job.log" "$C_RESET"
done
