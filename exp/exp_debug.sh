#!/bin/bash
trap "echo Terminated: $0; exit 1"  1 2 3 15

SCRIPT_BASENAME="$(basename ${0})"
SCRIPT_NAME=${SCRIPT_BASENAME%.*}
DATE="$(date "+%Y%m%d-%H%M%S")"
echo "============================================"
echo "${DATE} ${SCRIPT_NAME} START"
echo "============================================"

### CONFIG ###################################################################
EXP_VERSION='DEBUGGING'

### MAIN ######################################################################
#MAIN="../mains/ET_SHANNON_Multiclass.py"
MAIN='src/main/MBBTraining.py'
# MAIN='/Users/koki_k/Documents/learning/同志社大学/4回生_CCILAB/mbb_training_prmu/src/main/MBBTraining.py'
# MAIN='/Users/koki_ccilab/Documents/learning/同志社大学/4回生_CCILAB/mbb_training_prmu/src/main/MBBTraining.py'
MAIN_BASENAME="$(basename $MAIN)"
MAIN_NAME=${MAIN_BASENAME%.*}
MAIN_DASHNAME=${MAIN_NAME//_/\-}

### DATASET ###################################################################
# DATASET_BASEDIR="/Users/koki_ccilab/Documents/learning/同志社大学/4回生_CCILAB/mbb_training_prmu/dataset"
DATASET_BASEDIR="dataset"

DATASET=(
  # "GMM5C.csv" 
  # "German.csv" 
  # "Abalone01.csv"
  # "Abalone.csv" 
  # "BreastCancer.csv" 
  # "GMM2000.csv"
  # "GMM300.csv" 
  # "Cardiotocography2C.csv" 
  # "Thyroid2C.csv"
  # "WineRed3C.csv"
  # "WineWhite3C.csv" 
  # "Satimage.csv" 
  "Spambase.csv" 
  # "Cardiotocographic.csv"
  # "data_Tex.csv"
  # "data_Sha.csv"
  # "data_Mar.csv"
)

### EXPERIMENT PARAMETER ######################################################
EPOCH=10000
PROTO=100
KNN=40
LEARNING_RATE=(\
# #  5.0 \
#  1.0 \
#  0.25 \
 0.1 \
#  0.05 \
#  0.01 \ 
)
INTERVAL_NUM=1000
REEST_B_INTERVAL=$INTERVAL_NUM
REEST_P_INTERVAL=$INTERVAL_NUM
PLOT_INTERVAL=$INTERVAL_NUM

### RUN EXPERIMENT ###########################################################
RESULT_BASEDIR="./res/${MAIN_DASHNAME}/${EXP_VERSION}"
for ds in ${DATASET[@]}; do
  for lr in ${LEARNING_RATE[@]}; do

    LR_PNAME="LR${lr//./p}"

    DATASET_PATH="${DATASET_BASEDIR}/${ds}"
    DATASET_BASENAME="$(basename $DATASET_PATH)"
    DATASET_NAME=${DATASET_BASENAME%.*}
    DATASET_DASHNAME=${DATASET_NAME//_/\-}

    if [ "$DATASET_BASENAME" = "GMM2000.csv" ]; then
      TRAINING_SIZE=2000
      TESTING_SIZE=10000
    elif [ "$DATASET_BASENAME" = "GMM300.csv" ]; then
      TRAINING_SIZE=300
      TESTING_SIZE=11700
    else
      TRAINING_SIZE=0.5
      TESTING_SIZE=0.5
    fi

    RESULT_DIR="${RESULT_BASEDIR}/${DATASET_DASHNAME}/${DATE}_${LR_PNAME}"

    HYPER_PARAMETERS="\
    --name ${SCRIPT_NAME}-${DATASET_DASHNAME}
    --dataset $DATASET_BASEDIR/$ds \
    --training-size $TRAINING_SIZE \
    --testing-size $TESTING_SIZE \
    --output $RESULT_DIR \
    --plotting-interval $PLOT_INTERVAL \
    --epoch $EPOCH \
    --reest-B-interval $REEST_B_INTERVAL \
    --reest-P-interval $REEST_P_INTERVAL \
    --proto $PROTO \
    --learning-rate $lr \
    --knn-size $KNN \
    "
    # echo "python" $MAIN $HYPER_PARAMETERS
    # eval "python" $MAIN $HYPER_PARAMETERS
    echo "~/miniforge3/envs/cci/bin/python" $MAIN $HYPER_PARAMETERS
    eval "~/miniforge3/envs/cci/bin/python" $MAIN $HYPER_PARAMETERS
    # echo "~/opt/anaconda3/envs/tf/bin/python" $MAIN $HYPER_PARAMETERS
    # eval "~/opt/anaconda3/envs/tf/bin/python" $MAIN $HYPER_PARAMETERS

    # echo "python" $MAIN $HYPER_PARAMETERS
    # eval "python" $MAIN $HYPER_PARAMETERS
    wait
  done
done

DATE="$(date "+%Y%m%d-%H%M%S")"
echo "++++++++++++++++++++++++++++++++++++++++++++"
echo "${DATE} ${SCRIPT_NAME} DONE (${SECONDS} sec.)"
echo "++++++++++++++++++++++++++++++++++++++++++++"

