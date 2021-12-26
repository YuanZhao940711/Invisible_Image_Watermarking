# The "test" has two options: generate / extract
python3 main_watermarking.py \
  --imageSize 128 \
  --bs_secret 1 \
  --num_training 1 \
  --num_secret 1 \
  --num_cover 1 \
  --channel_cover 3 \
  --channel_secret 3 \
  --norm 'batch' \
  --loss 'l2' \
  --beta 0.75 \
  --test 'extract' \
  --bit_size 8 \
  --checkpoint './training/main_watermarking/checkPoints/best_checkpoint.pth.tar' \
  --secpath './covimg_randbits.jpg' \
  --cov_dir '/content/drive/MyDrive/FaceIDWatermarking/ffhq_rec/seq_weight_0.1_mls/original' \
  --con_dir '/content/drive/MyDrive/UniversalDeepHiding/experiment/container' \
  --outdir './exp'
