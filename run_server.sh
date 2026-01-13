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
    echo -e "${CYAN}║${NC}                                                              ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  ${YELLOW}FLUX.2 Models:${NC}                                             ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${GREEN}4)${NC} FLUX.2 4-bit BNB      (Low VRAM, local encoder)       ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${GREEN}5)${NC} FLUX.2 Full           (High VRAM, best quality)       ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${GREEN}6)${NC} FLUX.2 Full + Turbo   (8-step fast inference)         ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                              ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}    ${RED}q)${NC} Quit                                                  ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                              ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Start server with selected configuration
start_server() {
    local config=$1
    local args=""
    local desc=""

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
            args="--flux2"
            desc="FLUX.2 4-bit BNB"
            ;;
        5)
            args="--flux2 --full-model"
            desc="FLUX.2 Full"
            ;;
        6)
            args="--flux2 --full-model --turbo"
            desc="FLUX.2 Full + Turbo"
            ;;
        *)
            echo -e "${RED}Invalid selection${NC}"
            return 1
            ;;
    esac

    echo ""
    echo -e "${GREEN}Starting ${desc}...${NC}"
    echo -e "${BLUE}Command: python web_server.py $args${NC}"
    echo ""

    kill_existing_server

    source .venv/bin/activate

    # Save PID and start server
    echo $$ > server.pid
    python web_server.py $args "$@"
}

# Main loop
main() {
    # Check if a number was passed as argument
    if [ -n "$1" ] && [[ "$1" =~ ^[1-6]$ ]]; then
        start_server "$1" "${@:2}"
        exit $?
    fi

    while true; do
        show_menu
        echo -ne "${CYAN}Select configuration [1-6, q to quit]: ${NC}"
        read -r choice

        case $choice in
            [1-6])
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
