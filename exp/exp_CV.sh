#!/bin/bash
trap "echo Terminated: $0; exit 1"  1 2 3 15

SCRIPT_BASENAME="$(basename ${0})"
SCRIPT_NAME=${SCRIPT_BASENAME%.*}
DATE="$(date "+%Y%m%d-%H%M%S")"
echo "============================================"
echo "${DATE} ${SCRIPT_NAME} START"
echo "============================================"

### CONFIG ###################################################################
MAIN='src/main/CV.py'
# RESULT_BASEDIR="../res/${SCRIPT_NAME}"
RESULT_BASEDIR="res/${SCRIPT_NAME}"
mkdir -p "${RESULT_BASEDIR}"

### MAIN ######################################################################
DATASET_BASEDIR="dataset"
DATASET=(\
# "Abalone01.csv" \
# "Abalone.csv" \
# "BreastCancer.csv" \
# "Cardiotocography2C.csv" \
 "German.csv" \
# "GMM5C.csv" \
#  "GMM300.csv" \
#  "GMM2000.csv" \
#  "Satimage.csv" \
#  "Spambase.csv" \
#  "Thyroid2C.csv" \
#  "WineRed3C.csv" \
#  "WineWhite3C.csv" \
#"Avila.csv" \
#"Banknote.csv" \
#"Cardio.csv" \
#"Cardiotocography.csv" \
#"GMM12000.csv" \
#"Ionosphere.csv" \
#"Sonar.csv" \
#"Thyroid.csv" \
#  "data_Tex.csv"
#   "data_Sha.csv"
#   "data_Mar.csv"
)

# Prefix: m...minus value, p...plus value
MIN_PBC="p1"
MAX_PBC="p100"
MIN_SVM="m15"
MAX_SVM="p15"
FOLD=160

# MODE="ALL" or "PBC" or "SVM"
#MODE="PBC"
#MODE="SVM"
MODE="ALL"

for ds in ${DATASET[@]}; do
  echo "$ds"

  HYPER_PARAMETERS="\
  --dataset $DATASET_BASEDIR/$ds \
  --output $RESULT_BASEDIR \
  --min-p $MIN_PBC \
  --max-p $MAX_PBC \
  --min-s $MIN_SVM \
  --max-s $MAX_SVM \
  --mode $MODE \
  --fold $FOLD \
  "
  echo "~/miniforge3/envs/cci/bin/python" $MAIN $HYPER_PARAMETERS
  eval "~/miniforge3/envs/cci/bin/python" $MAIN $HYPER_PARAMETERS
  wait
done

DATE="$(date "+%Y%m%d-%H%M%S")"
echo "++++++++++++++++++++++++++++++++++++++++++++"
echo "${DATE} ${SCRIPT_NAME} DONE (${SECONDS} sec.)"
echo "++++++++++++++++++++++++++++++++++++++++++++"
