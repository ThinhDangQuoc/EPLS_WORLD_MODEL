#!/bin/bash
# EPLS Replication Run Scripts
# Cách dùng: bash run_epls.sh [phase]
# Ví dụ:
#   bash run_epls.sh 1   -> Phase 1: Benchmark baseline (World_Model_Random)
#   bash run_epls.sh 2   -> Phase 2: Iterative training (5 iterations)
#   bash run_epls.sh 3   -> Phase 3: Final benchmark (iterative model)

PHASE=${1:-"help"}
PYTHON=/home/thinh/anaconda3/bin/python
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"

run_headless() {
    local config=$1
    local log=$2
    echo ""
    echo "========================================"
    echo " Config: $config"
    echo " Log:    $log"
    echo " Time:   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"

    # Thử chạy với xvfb nếu có, không thì chạy thường
    if command -v xvfb-run &> /dev/null; then
        cp "$config" config.json
        xvfb-run -a -s "-screen 0 1280x1024x24" -- $PYTHON main.py 2>&1 | tee "$log"
    else
        echo "[WARN] xvfb-run không có, thử chạy trực tiếp..."
        cp "$config" config.json
        $PYTHON main.py 2>&1 | tee "$log"
    fi
}

cd "$PROJ_DIR"

case $PHASE in
    1)
        echo ">>> PHASE 1: Benchmark World_Model_Random (mục tiêu ~356 điểm)"
        echo ">>> Ước tính thời gian: 2-3 giờ (10 trials)"
        run_headless "config_phase1_benchmark.json" "log_phase1_benchmark.txt"
        ;;
    2)
        echo ">>> PHASE 2: Iterative Training - 5 iterations"
        echo ">>> Ước tính thời gian: 18-30 giờ"
        echo ""
        # Đảm bảo World_Model_Iter_A checkpoint tồn tại
        if [ ! -f "mdrnn/checkpoints/World_Model_Iter_A_mdrnn_best.tar" ]; then
            echo "Copying baseline model for Iter_A..."
            cp mdrnn/checkpoints/World_Model_Random_mdrnn_best.tar \
               mdrnn/checkpoints/World_Model_Iter_A_mdrnn_best.tar
        fi
        if [ ! -f "vae/checkpoints/World_Model_Iter_A_vae_best.tar" ]; then
            echo "Copying VAE for Iter_A..."
            cp vae/checkpoints/World_Model_Random_vae_best.tar \
               vae/checkpoints/World_Model_Iter_A_vae_best.tar
        fi
        run_headless "config_phase2_iterative.json" "log_phase2_iterative.txt"
        ;;
    3)
        echo ">>> PHASE 3: Final Benchmark - iterative_World_Model_Iter_A (mục tiêu 708 ± 195)"
        echo ">>> Ước tính thời gian: 2-3 giờ (100 trials)"
        run_headless "config_phase3_final_benchmark.json" "log_phase3_benchmark.txt"
        ;;
    *)
        echo ""
        echo "EPLS Replication Script"
        echo "Cách dùng: bash run_epls.sh [phase]"
        echo ""
        echo "  Phase 1: Benchmark World_Model_Random (~356 điểm, ~2-3 giờ)"
        echo "  Phase 2: Iterative Training 5 vòng (~708 điểm, ~18-30 giờ)"  
        echo "  Phase 3: Final benchmark 100 trials (~2-3 giờ)"
        echo ""
        echo "Models sẵn có:"
        ls mdrnn/checkpoints/ | grep -v backup | grep -v basemodel | sed 's/^/  /'
        ;;
esac
