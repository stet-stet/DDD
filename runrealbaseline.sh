python train_baseline.py \
  +augmentor=shift_only \
  +tr_loader=clippedclean \
  +cv_loader=clippedclean_5s \
  +ts_loader=setting_5s \
  +loss=setting_1 \
  +model=realbaseline \
  +optimizer=adamw_1e4 \
  +experiment=realbaseline \
  +data=vbdm_clip \
  +solver=audiotoaudio 