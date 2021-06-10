#! /bin/bash

function runmodel() {
    __NAME=0;
    __LOSS=0;   
    __EPOCH=64;
    __USEAUG=0;

    if [ x$1 != x ]; then
        __NAME=$1;       
    fi
    if [ x$2 != x ]; then
        __LOSS=$2;        
    fi
    if [ x$3 != x ]; then
        __EPOCH=$3;        
    fi
    if [ x$4 != x ]; then
        __USEAUG=$4;        
    fi

    echo  $(date +"[%Y/%m/%d %H:%M:%S]") " => Start Train && Test && Evaluate for" $__NAME;

    python -u train.py --name $__NAME \
        --save_model ./Snapshot/SINet_COD10K_AllCam_$__NAME \
        --train_img_dir ./Dataset/TrainDataset_AllCam/Image_COD10K/ \
        --train_gt_dir ./Dataset/TrainDataset_AllCam/GT_COD10K/ \
        --lossf $__LOSS \
        --max_epoch $__EPOCH \
        --use_aug $__USEAUG \
        >> $__NAME.train.log \
    && \
    python test.py --name $__NAME \
        --model_path ./Snapshot/SINet_COD10K_AllCam_$__NAME/SINet.pth \
        --test_save ./Result/SINet_COD10K_AllCam_$__NAME/ \
        --mask_root ./Dataset/TestDataset/ \
        --use_aug $__USEAUG \
    && \
    python test_metrics.py \
        --mask_root ./Dataset/TestDataset/COD10K_all_cam/GT \
        --pred_root ./Result/SINet_COD10K_AllCam_$__NAME/COD10K_all_cam \
    && \
    python test_metrics.py \
        --mask_root ./Dataset/TestDataset/CAMO/GT \
        --pred_root ./Result/SINet_COD10K_AllCam_$__NAME/CAMO \
    && \
    python test_metrics.py \
        --mask_root ./Dataset/TestDataset/CHAMELEON/GT \
        --pred_root ./Result/SINet_COD10K_AllCam_$__NAME/CHAMELEON \
    
}

function runmodelinf() {
    __NAME=0;
    if [ x$1 != x ]; then
        __NAME=$1;       
    fi

    echo  $(date +"[%Y/%m/%d %H:%M:%S]") " => Start Train && Test && Evaluate for" $__NAME;

    python -u train.py --name $__NAME \
        --save_model ./Snapshot/SINet_COD10K_AllCam_$__NAME \
        --train_img_dir ./Dataset/TrainDataset_AllCam/Image_COD10K/ \
        --train_gt_dir ./Dataset/TrainDataset_AllCam/GT_COD10K/ \
        --lossf 1 \
        --use_aug 1 \
        --max_epoch 300 \
        --min_epoch 100 \
        --save_epoch 20 \
        >> $__NAME.train.log \
    && \
    python test.py --name $__NAME \
        --model_path ./Snapshot/SINet_COD10K_AllCam_$__NAME/SINet.pth \
        --test_save ./Result/SINet_COD10K_AllCam_$__NAME/ \
        --mask_root ./Dataset/TestDataset/ \
        --use_aug 1 \
    && \
    python test_metrics.py \
        --mask_root ./Dataset/TestDataset/COD10K_all_cam/GT \
        --pred_root ./Result/SINet_COD10K_AllCam_$__NAME/COD10K_all_cam \
    && \
    python test_metrics.py \
        --mask_root ./Dataset/TestDataset/CAMO/GT \
        --pred_root ./Result/SINet_COD10K_AllCam_$__NAME/CAMO \
    && \
    python test_metrics.py \
        --mask_root ./Dataset/TestDataset/CHAMELEON/GT \
        --pred_root ./Result/SINet_COD10K_AllCam_$__NAME/CHAMELEON \
    
}

# nohup bash ./op.sh >> nohup.log &

# sleep 5; runmodel siv2 0 64 0
# sleep 5; runmodel sidecs 0 64 0

# sleep 5; runmodel sloss_v2 1 64 0
# sleep 5; runmodel sloss_decs 1 64 0

# sleep 5; runmodel sloss100_v2 1 100 0
# sleep 5; runmodel sloss100_decs 1 100 0

# sleep 5; runmodel sloss100aug_v2 1 100 1
# sleep 5; runmodel sloss100aug_decs 1 100 1

# sleep 5; runmodel bce100_v2 0 100 0
# sleep 5; runmodel bce100_decs 0 100 0

# sleep 5; runmodel bce100aug_v2 0 100 1
# sleep 5; runmodel bce100aug_decs 0 100 1

# sleep 5; runmodel Single_MinDecs2 1 100 1
# sleep 5; runmodel Single_MinDec2s2 1 100 1

# sleep 5; runmodel Single_MinDecs2 1 100 1
# sleep 5; runmodel Single_MinDec2s2 1 100 1

sleep 5; runmodelinf slossaug_decs_inf
sleep 5; runmodelinf slossaug_mindecs2_inf
sleep 5; runmodelinf slossaug_v2_inf




