#!/bin/bash
# x lower, x upper, y lower, y upper



# DIR='/home/robotsrule/Dropbox/control_variate_results/final_results'  # for submission
TOPDIR='/home/robotsrule/Dropbox/control_variate_results/final_results_arxiv'  # for arxiv

if [ $1 = "sigmas" ] || [ $1 = "all" ]; then
    # declare -a STDS=("01" "02" "04" "08")  # for submission
    declare -a STDS=("0-1" "0-3" "1-0" "3-0")  # for arxiv
    DIR="${TOPDIR}/sigmas"
    N_ITERS=30
    LEGEND_LOC=0
    for STD in "${STDS[@]}"; do
        python scripts/plot_sigmas.py --logdir ${DIR}/${STD}  --style corl_sigma --output ../std_${STD}.pdf --n_iters $N_ITERS --legend_loc $LEGEND_LOC --y_higher 30000 --y_lower 1 --legend_loc 4
    done
fi

if [ $1 = "cp" ] || [ $1 = "all" ]; then
    declare -a EXPS=("1k_mc" "2k_mc" "4k_mc")
    DIR="${TOPDIR}/cp"
    N_ITERS=50
    LEGEND_LOC=0
    for EXP in "${EXPS[@]}"; do
        python scripts/plot.py --logdir_parent ${DIR}/${EXP} --value MeanSumOfRewards --style corl_cp_arxiv --output ../${EXP}.pdf --n_iters $N_ITERS --legend_loc $LEGEND_LOC 
    done
fi



if [ $1 = "cp-quad" ] || [ $1 = "all" ]; then
    declare -a EXPS=("4k_quad_diff" "4k_quad_next")
    DIR="${TOPDIR}/cp"
    N_ITERS=50
    LEGEND_LOC=0
    for EXP in "${EXPS[@]}"; do
        python scripts/plot.py --logdir_parent ${DIR}/${EXP} --value MeanSumOfRewards --style corl_cp_arxiv --output ../${EXP}.pdf --n_iters $N_ITERS --legend_loc $LEGEND_LOC 
    done
fi

