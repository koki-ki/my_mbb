#!/bin/bash
trap "echo Terminated: $0; exit 1"  1 2 3 15

SCRIPT_BASENAME="$(basename ${0})"
SCRIPT_NAME=${SCRIPT_BASENAME%.*}
DATE="$(date "+%Y%m%d-%H%M%S")"
echo "============================================"
echo "${DATE} ${SCRIPT_NAME} START"
echo "============================================"

### CONFIG ###################################################################
GIT_VERSION=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
EXP_VERSION=${GIT_VERSION:="DEBUGGING"}

### MAIN ######################################################################
#MAIN="../mains/ET_SHANNON_Multiclass.py"
MAIN='../src/main/MBBTraining.py'
MAIN_BASENAME="$(basename $MAIN)"
MAIN_NAME=${MAIN_BASENAME%.*}
MAIN_DASHNAME=${MAIN_NAME//_/\-}

### DATASET ###################################################################
DATASET_BASEDIR="../dataset"
DATASET=(\
  # "GMM5C.csv" \
  #  "German.csv" \
    "Abalone01.csv" 
  # "Abalone.csv" \
  # "BreastCancer.csv" \
  # "GMM2000.csv" \
  # "GMM300.csv" \
  # "Cardiotocography2C.csv" \
  # "Thyroid2C.csv" \
  # "WineRed3C.csv" \
  # "WineWhite3C.csv" \
  # "Satimage.csv" \
  # "Spambase.csv" \
  #  "data_Tex.csv"
  # "data_Sha.csv"
  # "data_Mar.csv"
)

### EXPERIMENT PARAMETER ######################################################
EPOCH=10000
PROTO=(\
  300 \
  200 \
  150 \
#  100 \
#  80 \
#  60 \
#  40 \
#  20 \
#  18 \
#  16 \
#  14 \
#  12 \
#  10 \
#  8 \
#  6 \
  #  50 \
  #  250 \
  #  500 \
  # 20 \
  # 18 \
  # 16 \
  # 14 \
  # 12 \
  # 8 \
  # 6 \
  # 2 \
  # 4 \
  )
KNN=(\
  # 20 \
  40 \
  # 60 \
  # 80 \
  # 100 \
  )
LEARNING_RATE=(\
  #  10.0 \
  #  5.0 \
  # 1.0 \
  # 0.5 \
  # 0.1 \
  # 0.05 \
    0.01 \
)
INTERVAL_NUM=1000
REEST_B_INTERVAL=$INTERVAL_NUM
REEST_P_INTERVAL=$INTERVAL_NUM
PLOT_INTERVAL=$INTERVAL_NUM

### RUN EXPERIMENT ###########################################################
RESULT_BASEDIR="../result3/${MAIN_DASHNAME}/${EXP_VERSION}"
for ds in ${DATASET[@]}; do
  for lr in ${LEARNING_RATE[@]}; do
    for pt in ${PROTO[@]}; do
       for knn in ${KNN[@]}; do
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

          RESULT_DIR="${RESULT_BASEDIR}/${DATASET_DASHNAME}/${DATE}_${LR_PNAME}_${pt}"

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
          --proto $pt \
          --learning-rate $lr \
          --knn-size $knn \
          "

          echo "python" $MAIN $HYPER_PARAMETERS
          eval "python" $MAIN $HYPER_PARAMETERS
          wait
      done
    done
  done
done
DATE="$(date "+%Y%m%d-%H%M%S")"
echo "++++++++++++++++++++++++++++++++++++++++++++"
echo "${DATE} ${SCRIPT_NAME} DONE (${SECONDS} sec.)"
echo "++++++++++++++++++++++++++++++++++++++++++++"
