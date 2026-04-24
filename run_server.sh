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
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}           ${GREEN}FLUX Image Generator - Server Launcher${NC}            ${CYAN}║${NC}"
    echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC}                                                              ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  ${YELLOW}FLUX.1 Models:${NC}                                             ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${GREEN}1)${NC} FLUX.1 4-bit BNB      (Low VRAM, remote encoder)      ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${GREEN}2)${NC} FLUX.1 Full           (High VRAM, best quality)       ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${GREEN}3)${NC} FLUX.1 GGUF Q8        (DGX Spark optimized)           ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${GREEN}4)${NC} FLUX.1-schnell        (4-step fast, Apache 2.0)       ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${GREEN}8)${NC} FLUX.1 + Uncensored   (Full model + LoRA)             ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                              ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  ${YELLOW}FLUX.2 Models:${NC}                                             ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${GREEN}5)${NC} FLUX.2 4-bit BNB      (Low VRAM, local encoder)       ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${GREEN}6)${NC} FLUX.2 Full           (High VRAM, best quality)       ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${GREEN}7)${NC} FLUX.2 Full + Turbo   (8-step fast inference)         ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${GREEN}9)${NC} FLUX.2 Full (no Turbo) (Max quality, slower)          ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${GREEN}10)${NC} FLUX.2-klein-9B       ${YELLOW}[default]${NC} (9B, faster)          ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                              ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${RED}q)${NC} Quit                                                  ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                              ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Start server with selected configuration (with auto-restart on failure)
start_server() {
    local config=$1
    shift  # Remove config number from args
    local args=""
    local desc=""
    local max_retries=5
    local retry_delay=3
    local log_file="server.log"

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
            args="--flux2"
            desc="FLUX.2 4-bit BNB"
            ;;
        6)
            args="--flux2 --full-model"
            desc="FLUX.2 Full"
            ;;
        7)
            args="--flux2 --full-model --turbo"
            desc="FLUX.2 Full + Turbo"
            ;;
        8)
            args="--uncensored"
            desc="FLUX.1 + Uncensored LoRA"
            ;;
        9)
            args="--flux2 --full-model --no-turbo"
            desc="FLUX.2 Full (no Turbo)"
            ;;
        10)
            args="--klein"
            desc="FLUX.2-klein-9B"
            ;;
        *)
            echo -e "${RED}Invalid selection${NC}"
            return 1
            ;;
    esac

    echo ""
    echo -e "${GREEN}Starting ${desc}...${NC}"
    echo -e "${BLUE}Command: python web_server.py $args${NC}"
    echo -e "${YELLOW}Auto-restart enabled (max $max_retries retries on failure)${NC}"
    echo -e "${CYAN}Logging to: $log_file${NC}"
    echo ""

    kill_existing_server

    source .venv/bin/activate

    # Save parent PID
    echo $$ > server.pid

    # Restart loop
    local attempt=0
    while true; do
        attempt=$((attempt + 1))
        local start_time=$(date +%s)

        if [ $attempt -gt 1 ]; then
            echo "" | tee -a "$log_file"
            echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}" | tee -a "$log_file"
            echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] Restart attempt $attempt of $max_retries${NC}" | tee -a "$log_file"
            echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}" | tee -a "$log_file"
            echo "" | tee -a "$log_file"
        else
            echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] Initial server start${NC}" | tee -a "$log_file"
        fi

        # Run the server and append output to log
        # We use 'tee -a' for the script's own messages, but we want the python output 
        # to go to the log file. We also want to see it in the terminal.
        python web_server.py $args "$@" 2>&1 | tee -a "$log_file"
        exit_code=${PIPESTATUS[0]}

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

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

    rm -f server.pid
}

# Main loop
main() {
    # Check if a number was passed as argument
    if [ -n "$1" ] && [[ "$1" =~ ^([1-9]|10)$ ]]; then
        start_server "$@"
        exit $?
    fi

    # No arguments passed → launch default (klein) directly
    if [ -z "$1" ] && [ ! -t 0 ]; then
        start_server 10
        exit $?
    fi

    while true; do
        show_menu
        echo -ne "${CYAN}Select configuration [1-10, default=10, q to quit]: ${NC}"
        read -r choice
        # Empty input → run default (klein)
        if [ -z "$choice" ]; then
            choice=10
        fi

        case $choice in
            [1-9]|10)
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
