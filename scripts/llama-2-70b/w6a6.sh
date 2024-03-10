SAVE_PATH=save_path
python main.py --model model_path \
--wbits 6 \
--abits 6 \
--eval_ppl \
--use_lora \
--output_dir ${SAVE_PATH} \
--lr 1e-4 \
--num_layer 4 \
--epochs 10 \
--plot_act_max \
--channel_ratio 0.2 \
--plot_num_additional_channels \
--calibrate_bs 1 \
--num_gpu 1 \
--nsamples 128 \
--batch_size 1