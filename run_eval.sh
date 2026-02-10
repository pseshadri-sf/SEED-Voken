python evaluation_image.py \
--config_file configs/Open-MAGVIT2/gpu/eval_ejepa_128.yaml \
--ckpt_path /root/workspace/SEED-Voken/checkpoints/vqgan/opencua/epoch=24-step=257625.ckpt  \
--model Open-MAGVIT2  \
--image_size 128  \
--batch_size 4 \
--save_comparison_dir ./eval_comparisons \
--save_native_resolution