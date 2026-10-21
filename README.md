# Environment-Aware-Channel-Inference-via-Physics-Informed-Flow-Matching

Use the following script to run the PBFM training. Change the data-root to Town 3's actual location in memory.

python Flow_Matching_Train_MMW_Nopilot.py \
  --data_root "/home/name/datasets/Multimodal-Wireless" \
  --stage fm \
  --physics_loss true \
  --vae_checkpoint_dir outputs \
  --flow_checkpoint outputs/flow_matching_mmw_best.pth \
  --output_dir outputs_pbfm_town03 \
  --epochs 2000 \
  --batch_size 64 \
  --physics_unroll_max_steps 4 \
  --num_workers 4
