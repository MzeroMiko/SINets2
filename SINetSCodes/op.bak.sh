# Dataset Folder

# Dataset/TestDataset
# ├── CAMO
# │   ├── Edge
# │   ├── GT
# │   └── Imgs
# ├── CHAMELEON
# │   ├── Edge
# │   ├── GT
# │   └── Imgs
# ├── COD10K
# │   ├── GT
# │   ├── GT_Edge
# │   ├── GT_Instance
# │   └── Imgs
# └── COD10K_all_cam
#     ├── Edge
#     ├── GT
#     └── Imgs
# Dataset/TrainDataset_AllCam
# ├── GT
# ├── GT_COD10K
# ├── Image
# └── Image_COD10K

# --------------------------------------------------------------------------------

# To Train Ori
__NAME="SINet_ResNet50_Ori" && \
nohup python -u train_ori.py --name $__NAME \
    --save_model ./Snapshot/SINet_COD10K_AllCam_$__NAME \
    --train_img_dir ./Dataset/TrainDataset_AllCam/Image_COD10K/ \
    --train_gt_dir ./Dataset/TrainDataset_AllCam/GT_COD10K/ \
    >> $__NAME.train.log &

# To Test Ori
__NAME="SINet_ResNet50_Ori" && \
nohup python -u test_ori.py --name $__NAME \
    --model_path ./Snapshot/SINet_COD10K_AllCam_$__NAME/SINet_40.pth \
    --test_save ./Result/SINet_COD10K_AllCam_$__NAME/ \
    --mask_root ./Dataset/TestDataset/ &

# --------------------------------------------------------------------------------

# To Train
__NAME="SINet_SimpF1" && \
nohup python -u train.py --name $__NAME \
    --save_model ./Snapshot/SINet_COD10K_AllCam_$__NAME \
    --train_img_dir ./Dataset/TrainDataset_AllCam/Image_COD10K/ \
    --train_gt_dir ./Dataset/TrainDataset_AllCam/GT_COD10K/ \
    >> $__NAME.train.log &

# To Test
__NAME="SINet_RFDCTATT" && \
nohup python -u test.py --name $__NAME \
    --model_path ./Snapshot/SINet_COD10K_AllCam_$__NAME/SINet.pth \
    --test_save ./Result/SINet_COD10K_AllCam_$__NAME/ \
    --mask_root ./Dataset/TestDataset/ &

# ---------------------------------------------------------------------------

# To Test Metric
__NAME="SINet_RFDCTATT" && \
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
# && \
# python test_metrics.py \
#     --mask_root ./Dataset/TestDataset/COD10K/GT \
#     --pred_root ./Result/SINet_COD10K_AllCam_$__NAME/COD10K \

