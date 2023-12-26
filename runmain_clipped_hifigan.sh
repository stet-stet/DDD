python train_hifigan.py \
  +augmentor=shift_only \
  +tr_loader=clippedclean \
  +cv_loader=clippedclean \
  +ts_loader=setting_1 \
  +loss=setting_1 \
  +model=setting_1 \
  +optimizer=adamw_1e4 \
  +experiment=setting_3_bs2 \
  +data=vbdm_clip \
  +solver=hifiganaudiotoaudio 