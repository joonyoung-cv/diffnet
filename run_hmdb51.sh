#! /bin/bash
# HMDB-51 SPLIT 1.
# Epoch size is set to 4-times larger than the default.
DATA_OUT_DIR=/home/dgyoo/workspace/dataout/diffnet/HMDB51_RGB_S1

# Spatial net.
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 512 -epochSize 75 -numEpoch 30'
th main.lua -task vi_vggmpt_s_slcls $ITERATION -data HMDB51_RGB_S1
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 256 -epochSize 150 -numEpoch 30'
th main.lua -task vi_vgg16pt_s_slcls $ITERATION -data HMDB51_RGB_S1

# Diff net.
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 512 -epochSize 150 -numEpoch 30'
th main.lua -task vi_vggmpt_d_slcls $ITERATION -stride 4 -diffLevel 1 -data HMDB51_RGB_S1
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 256 -epochSize 299 -numEpoch 20'
th main.lua -task vi_vgg16pt_d_slcls $ITERATION -stride 4 -diffLevel 4 -data HMDB51_RGB_S1

# Spatial+diff net.
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 512 -epochSize 150 -numEpoch 30'
th main.lua -task vi_vggmpt_sdsel_slcls $ITERATION -stride 4 -branchAfter 1 -mergeAfter 5 -data HMDB51_RGB_S1
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 192 -epochSize 398 -numEpoch 30'
th main.lua -task vi_vgg16pt_sdsel_slcls $ITERATION -stride 4 -branchAfter 4 -mergeAfter 13 -data HMDB51_RGB_S1


# HMDB-51 SPLIT 2.
# Epoch size is set to 4-times larger than the default.
DATA_OUT_DIR=/home/dgyoo/workspace/dataout/diffnet/HMDB51_RGB_S2

# Spatial net.
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 512 -epochSize 75 -numEpoch 30'
th main.lua -task vi_vggmpt_s_slcls $ITERATION -data HMDB51_RGB_S2
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 256 -epochSize 150 -numEpoch 30'
th main.lua -task vi_vgg16pt_s_slcls $ITERATION -data HMDB51_RGB_S2

# Diff net.
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 512 -epochSize 150 -numEpoch 30'
th main.lua -task vi_vggmpt_d_slcls $ITERATION -stride 4 -diffLevel 1 -data HMDB51_RGB_S2
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 256 -epochSize 299 -numEpoch 20'
th main.lua -task vi_vgg16pt_d_slcls $ITERATION -stride 4 -diffLevel 4 -data HMDB51_RGB_S2

# Spatial+diff net.
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 512 -epochSize 150 -numEpoch 30'
th main.lua -task vi_vggmpt_sdsel_slcls $ITERATION -stride 4 -branchAfter 1 -mergeAfter 5 -data HMDB51_RGB_S2
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 192 -epochSize 398 -numEpoch 30'
th main.lua -task vi_vgg16pt_sdsel_slcls $ITERATION -stride 4 -branchAfter 4 -mergeAfter 13 -data HMDB51_RGB_S2


# HMDB-51 SPLIT 3.
# Epoch size is set to 4-times larger than the default.
DATA_OUT_DIR=/home/dgyoo/workspace/dataout/diffnet/HMDB51_RGB_S3

# Spatial net.
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 512 -epochSize 75 -numEpoch 30'
th main.lua -task vi_vggmpt_s_slcls $ITERATION -data HMDB51_RGB_S3
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 256 -epochSize 150 -numEpoch 30'
th main.lua -task vi_vgg16pt_s_slcls $ITERATION -data HMDB51_RGB_S3

# Diff net.
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 512 -epochSize 150 -numEpoch 30'
th main.lua -task vi_vggmpt_d_slcls $ITERATION -stride 4 -diffLevel 1 -data HMDB51_RGB_S3
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 256 -epochSize 299 -numEpoch 20'
th main.lua -task vi_vgg16pt_d_slcls $ITERATION -stride 4 -diffLevel 4 -data HMDB51_RGB_S3

# Spatial+diff net.
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 512 -epochSize 150 -numEpoch 30'
th main.lua -task vi_vggmpt_sdsel_slcls $ITERATION -stride 4 -branchAfter 1 -mergeAfter 5 -data HMDB51_RGB_S3
CUDA_VISIBLE_DEVICES=0,1,2,3
ITERATION='-batchSize 192 -epochSize 398 -numEpoch 30'
th main.lua -task vi_vgg16pt_sdsel_slcls $ITERATION -stride 4 -branchAfter 4 -mergeAfter 13 -data HMDB51_RGB_S3
