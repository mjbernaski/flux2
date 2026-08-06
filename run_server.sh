#!/bin/bash
# FLUX Server Launcher - Interactive menu to select model configuration

cd "$(dirname "$0")"

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check for and kill any existing server process
kill_existing_server() {
    if [ -f server.pid ]; then
        OLD_PID=$(cat server.pid)
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo -e "${YELLOW}Killing existing server (PID $OLD_PID)...${NC}"
            kill "$OLD_PID"
            sleep 1
        fi
        rm -f server.pid
    fi

    EXISTING=$(pgrep -f "python.*web_server.py" 2>/dev/null)
    if [ -n "$EXISTING" ]; then
        echo -e "${YELLOW}Killing existing web_server.py processes: $EXISTING${NC}"
        pkill -f "python.*web_server.py"
        sleep 1
    fi
}

# Display menu
show_menu() {
    clear
    echo -e "${CYAN}══════════════════════════════════════════════════${NC}"
    echo -e "       ${GREEN}FLUX Image Generator — Server Launcher${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${YELLOW}FLUX.1 (12B)${NC}"
    echo -e "    ${GREEN}1)${NC} 4-bit            Low VRAM"
    echo -e "    ${GREEN}2)${NC} Full             Best quality"
    echo -e "    ${GREEN}3)${NC} GGUF Q8          DGX Spark optimized"
    echo -e "    ${GREEN}4)${NC} schnell          4-step fast (Apache 2.0)"
    echo -e "    ${GREEN}5)${NC} U-LoRA           Full model + LoRA"
    echo ""
    echo -e "  ${YELLOW}FLUX.2 (32B · klein 9B/4B)${NC}"
    echo -e "    ${GREEN}6)${NC} 4-bit            Low VRAM"
    echo -e "    ${GREEN}7)${NC} Full (Turbo)     8-step fast inference"
    echo -e "    ${GREEN}8)${NC} Full (no Turbo)  Max quality, slower"
    echo -e "    ${GREEN}9)${NC} klein-9B         ${YELLOW}[default]${NC} faster 9B"
    echo -e "   ${GREEN}14)${NC} klein-4B         Lowest-VRAM FLUX.2, ~13GB bf16"
    echo ""
    echo -e "  ${YELLOW}Editing${NC}"
    echo -e "   ${GREEN}10)${NC} Kontext          Instruction editing (FLUX.1, 4-bit)"
    echo -e "   ${GREEN}11)${NC} Kontext Full     Instruction editing (FLUX.1, full bf16)"
    echo -e "   ${GREEN}12)${NC} Kontext U-LoRA   Kontext Full + U-LoRA (edit refs)"
    echo ""
    echo -e "  ${YELLOW}Stable Diffusion (SDXL)${NC}"
    echo -e "   ${GREEN}13)${NC} SDXL photoreal   Photoreal checkpoint, negative prompts"
    echo ""
    echo -e "    ${RED}q)${NC} Quit"
    echo ""
}

# Map a config number to server flags. Assigns the caller's `args` and `desc`
# (bash dynamic scoping). Keep in sync with SERVER_OPTIONS.md and the
# SERVER_CONFIGS table in web_server.py.
set_config_args() {
    local config=$1
    case $config in
        1)
            args=""
            desc="FLUX.1 4-bit BNB"
            ;;
        2)
            args="--full-model"
            desc="FLUX.1 Full"
            ;;
        3)
            args="--gguf q8 --local-encoder"
            desc="FLUX.1 GGUF Q8"
            # DGX Spark optimizations
            export TORCH_CUDA_ARCH_LIST="12.1"
            export CUDA_LAUNCH_BLOCKING=0
            ;;
        4)
            args="--schnell --local-encoder"
            desc="FLUX.1-schnell"
            ;;
        5)
            args="--uncensored"
            desc="FLUX.1 + U-LoRA"
            ;;
        6)
            args="--flux2"
            desc="FLUX.2 4-bit BNB"
            ;;
        7)
            args="--flux2 --full-model --turbo"
            desc="FLUX.2 Full + Turbo"
            ;;
        8)
            args="--flux2 --full-model --no-turbo"
            desc="FLUX.2 Full (no Turbo)"
            ;;
        9)
            args="--klein"
            desc="FLUX.2-klein-9B"
            ;;
        10)
            args="--kontext"
            desc="FLUX.1 Kontext (editor)"
            ;;
        11)
            args="--kontext --full-model"
            desc="FLUX.1 Kontext Full (editor, bf16)"
            ;;
        12)
            args="--kontext --full-model --uncensored"
            desc="FLUX.1 Kontext Full + U-LoRA"
            ;;
        13)
            # SDXL checkpoint (sd_core.py backend). Override the
            # checkpoint with SD_MODEL=<repo-or-path> before launching.
            args="--sdxl"
            desc="SDXL (photoreal)"
            ;;
        14)
            args="--klein-4b"
            desc="FLUX.2-klein-4B"
            ;;
        *)
            echo -e "${RED}Invalid selection${NC}"
            return 1
            ;;
    esac
}

# The critique/describe VLM needs a local ollama on 11434. The systemd unit
# is no use here: it runs as the `ollama` user, whose model store doesn't
# have qwen3.6 — the models live under ~/.ollama. So start `ollama serve` as
# this user, detached (setsid nohup) so it survives the SSH session ending.
ensure_ollama() {
    if ! command -v ollama >/dev/null 2>&1; then
        echo -e "${YELLOW}ollama not installed — VLM critique/describe will be unavailable${NC}"
        return 0
    fi
    if curl -sf --max-time 2 "http://127.0.0.1:11434/api/version" >/dev/null 2>&1; then
        return 0
    fi
    echo -e "${CYAN}Starting ollama (VLM backend, logging to ollama.log)...${NC}"
    setsid nohup ollama serve >> ollama.log 2>&1 < /dev/null &
    local i
    for i in $(seq 1 20); do
        if curl -sf --max-time 1 "http://127.0.0.1:11434/api/version" >/dev/null 2>&1; then
            echo -e "${GREEN}ollama is up.${NC}"
            return 0
        fi
        sleep 0.5
    done
    echo -e "${YELLOW}ollama did not come up within 10s — VLM features may be unavailable (see ollama.log)${NC}"
}

# web_server.py exits with this code (after writing .next_config) when the
# user picks a different model in the UI; the restart loop below relaunches
# with the new config instead of treating it as a crash.
SWITCH_EXIT_CODE=86
SWITCH_CONFIG_FILE=".next_config"

# Every (re)launch records its config number here, so `./run_server.sh last`
# — used by the flux-server systemd unit at boot — resumes whatever model
# was running before the reboot, including UI model switches.
LAST_CONFIG_FILE=".last_config"

# Start server with selected configuration (with auto-restart on failure)
start_server() {
    local config=$1
    shift  # Remove config number from args
    local args=""
    local desc=""
    local max_retries=5
    local retry_delay=3
    local log_file="server.log"

    set_config_args "$config" || return 1

    echo ""
    echo -e "${GREEN}Starting ${desc}...${NC}"
    echo -e "${BLUE}Command: python web_server.py $args${NC}"
    echo -e "${YELLOW}Auto-restart enabled (max $max_retries retries on failure)${NC}"
    echo -e "${CYAN}Logging to: $log_file${NC}"
    echo ""

    kill_existing_server
    ensure_ollama

    if [ ! -f .venv/bin/activate ]; then
        echo -e "${RED}No .venv found — create it first: python -m venv .venv && uv pip install -r requirements.txt${NC}"
        return 1
    fi
    source .venv/bin/activate

    # Save parent PID; the trap cleans the pidfile up on any exit, including
    # the max-retries give-up path and the script being killed.
    echo $$ > server.pid
    trap 'rm -f server.pid' EXIT

    # Restart loop
    local attempt=0
    local first_start=1
    while true; do
        attempt=$((attempt + 1))
        local start_time=$(date +%s)

        if [ $first_start -eq 0 ]; then
            echo "" | tee -a "$log_file"
            echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}" | tee -a "$log_file"
            echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] Restart attempt $attempt of $max_retries${NC}" | tee -a "$log_file"
            echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}" | tee -a "$log_file"
            echo "" | tee -a "$log_file"
        else
            first_start=0
            echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] Initial server start${NC}" | tee -a "$log_file"
        fi

        # Run the server and append output to log
        # We use 'tee -a' for the script's own messages, but we want the python output 
        # to go to the log file. We also want to see it in the terminal.
        # -u: unbuffered stdout so print() diagnostics land in the log/terminal
        # immediately (block-buffering through the tee pipe delays them by KBs).
        # FLUX_CONFIG tells the server which menu entry it is, enabling the
        # UI's on-the-fly model switcher (/configs + /switch-model).
        echo "$config" > "$LAST_CONFIG_FILE"
        FLUX_CONFIG=$config python -u web_server.py $args "$@" 2>&1 | tee -a "$log_file"
        exit_code=${PIPESTATUS[0]}

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        # Model switch requested from the web UI: relaunch with the new
        # config's flags. Not a crash — reset the retry counter.
        if [ $exit_code -eq $SWITCH_EXIT_CODE ] && [ -f "$SWITCH_CONFIG_FILE" ]; then
            local new_config
            new_config=$(cat "$SWITCH_CONFIG_FILE")
            rm -f "$SWITCH_CONFIG_FILE"
            if [[ "$new_config" =~ ^([1-9]|1[0-4])$ ]] && set_config_args "$new_config"; then
                config=$new_config
            else
                echo -e "${RED}Invalid switch request '$new_config' — restarting current model${NC}" | tee -a "$log_file"
                set_config_args "$config"
            fi
            echo "" | tee -a "$log_file"
            echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}" | tee -a "$log_file"
            echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')] Model switch — relaunching as: ${desc}${NC}" | tee -a "$log_file"
            echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}" | tee -a "$log_file"
            attempt=0
            first_start=1
            continue
        fi

        # Check exit code
        if [ $exit_code -eq 0 ]; then
            # Clean exit (user requested shutdown)
            echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] Server exited cleanly.${NC}" | tee -a "$log_file"
            break
        elif [ $exit_code -eq 130 ]; then
            # SIGINT (Ctrl+C) - user requested stop
            echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] Server stopped by user (Ctrl+C).${NC}" | tee -a "$log_file"
            break
        else
            # Process failed (includes exit code 137 from OOM killer)
            echo "" | tee -a "$log_file"
            echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}" | tee -a "$log_file"
            if [ $exit_code -eq 137 ]; then
                echo -e "${RED}║  [$(date '+%Y-%m-%d %H:%M:%S')] Server was killed (exit 137 - likely OOM)${NC}" | tee -a "$log_file"
            else
                echo -e "${RED}║  [$(date '+%Y-%m-%d %H:%M:%S')] Server crashed with exit code $exit_code${NC}" | tee -a "$log_file"
            fi

            # If the server was running for more than 60 seconds, reset the attempt counter
            if [ $duration -gt 60 ]; then
                echo -e "${GREEN}║  Server ran for $duration seconds. Resetting retry counter.${NC}" | tee -a "$log_file"
                attempt=0
            fi

            if [ $attempt -ge $max_retries ] && [ $attempt -ne 0 ]; then
                echo -e "${RED}║  Maximum retries ($max_retries) reached. Giving up.${NC}" | tee -a "$log_file"
                echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}" | tee -a "$log_file"
                return 1
            fi

            local next_attempt=$((attempt + 1))
            echo -e "${RED}║  Restarting in $retry_delay seconds... (attempt $next_attempt/$max_retries)${NC}" | tee -a "$log_file"
            echo -e "${RED}║  Press Ctrl+C to abort restart${NC}" | tee -a "$log_file"
            echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}" | tee -a "$log_file"

            # Wait with countdown, allowing Ctrl+C to cancel
            for i in $(seq $retry_delay -1 1); do
                echo -ne "\r${YELLOW}Restarting in $i...${NC}  "
                sleep 1
            done
            echo ""
        fi
    done
}

# Main loop
main() {
    # `last` → resume the most recently run config (falling back to 9/klein).
    # This is what the flux-server systemd unit passes at boot.
    if [ "$1" = "last" ]; then
        local last_config
        last_config=$(cat "$LAST_CONFIG_FILE" 2>/dev/null)
        if ! [[ "$last_config" =~ ^([1-9]|1[0-4])$ ]]; then
            last_config=9
        fi
        shift
        start_server "$last_config" "$@"
        exit $?
    fi

    # Check if a number was passed as argument
    if [ -n "$1" ] && [[ "$1" =~ ^([1-9]|1[0-4])$ ]]; then
        start_server "$@"
        exit $?
    fi

    # No arguments passed → launch default (klein) directly
    if [ -z "$1" ] && [ ! -t 0 ]; then
        start_server 9
        exit $?
    fi

    while true; do
        show_menu
        echo -ne "${CYAN}Select configuration [1-14, default=9, q to quit]: ${NC}"
        read -r choice
        # Empty input → run default (klein)
        if [ -z "$choice" ]; then
            choice=9
        fi

        case $choice in
            [1-9]|1[0-4])
                start_server "$choice"
                exit $?
                ;;
            q|Q)
                echo -e "${GREEN}Goodbye!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid selection. Press Enter to continue...${NC}"
                read -r
                ;;
        esac
    done
}

main "$@"
