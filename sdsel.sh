#! /bin/bash
# Epoch size is set to 4-times larger than the default.
DATA_OUT_DIR=/home/dgyoo/workspace/dataout/diffnet/UCF101_RGB_S1
DB_NAME=UCF101_RGB_S1
TASK=vi_vgg16pt_sdsel_slcls

# Train.
OPTION_TRAIN='-batchSize 192 -epochSize 398 -numEpoch 30'
OPTION_MODEL='-branchAfter 4 -mergeAfter 13 -stride 4 -diffChance 0.9'
CUDA_VISIBLE_DEVICES=0,1,2,3
th main.lua -task $TASK $OPTION_TRAIN $OPTION_MODEL -data $DB_NAME


# Test.
OPTION_TEST='-batchSize 192 -numGpu 2'
MODEL_DIR=$DATA_OUT_DIR/vi_vgg16pt_sdsel_slcls,batchSize=192,diffChance=0.9,epochSize=398,stride=4
CUDA_VISIBLE_DEVICES=0,1
for e in {1..30}
do
	START_FROM=$MODEL_DIR/$(printf 'model_%03d.t7' "$e")
	th testonly.lua -task $TASK $OPTION_TEST $OPTIOIN_MODEL -data $DB_NAME -startFrom $START_FROM
done
