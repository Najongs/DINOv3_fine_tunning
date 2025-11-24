torchrun --nproc_per_node=4 Single_view_3D_Loss.py --ablation_mode dino_conv_only
torchrun --nproc_per_node=4 Single_view_3D_Loss.py --ablation_mode siglip2_only
torchrun --nproc_per_node=4 Single_view_3D_Loss.py --ablation_mode dino_only

# Speed test
# python benchmark_resolution_speed.py \
#     --model-type dino_conv_only \
#     --checkpoint checkpoints_total_dino_conv_only/best_model.pth \
#     --combo 512x512:224x224 --combo 1280x720:1280x720 \
#     --device cuda:2 --iters 20

# python benchmark_resolution_speed.py \
#     --model-type dino_conv_only \
#     --checkpoint checkpoints_total_dino_conv_only/best_model.pth \
#     --combo 512x512:224x224 --combo 1280x720:1280x720 \
#     --device cuda:1 --iters 20