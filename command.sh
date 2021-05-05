##对比方法：
#1.baseline:(finished)
nohup python train_baseline.py --gpu 0 --model vgg19 --outdir save_baseline_test > baseline_vgg19.out 2>&1 &
nohup python train_baseline.py --gpu 1 --model resnet32 --outdir save_baseline_test > baseline_resnet32.out 2>&1 &
nohup python train_baseline.py --gpu 2 --model wide_resnet20_8 --outdir save_baseline_test > baseline_wide_resnet20_8.out 2>&1 &

nohup python train_baseline.py --gpu 2 --model vgg19 --outdir save_baseline2 > baseline_vgg19.out 2>&1 &
nohup python train_baseline.py --gpu 1 --model resnet32 --outdir save_baseline2 > baseline_resnet32.out 2>&1 &
nohup python train_baseline.py --gpu 3 --model wide_resnet20_8 --outdir save_baseline2 > baseline_wide_resnet20_8.out 2>&1 &

nohup python train_baseline.py --gpu 2 --model vgg19 --outdir save_baseline3 > baseline_vgg19.out 2>&1 &
nohup python train_baseline.py --gpu 2 --model resnet32 --outdir save_baseline3 > baseline_resnet32.out 2>&1 &
nohup python train_baseline.py --gpu 3 --model wide_resnet20_8 --outdir save_baseline3 > baseline_wide_resnet20_8.out 2>&1 &

#2.LWR:(finished)
nohup python train_LWR.py --gpu 1 --model vgg19 > LWR_vgg19.out 2>&1 &
nohup python train_LWR.py --gpu 1 --model resnet32 > LWR_resnet32.out 2>&1 &
nohup python train_LWR.py --gpu 2 --model wide_resnet20_8 > LWR_wide_resnet20_8.out 2>&1 &

#3.ban(finished 可以增加lr调整的trick)
nohup python train_ban.py --gpu 3 --model vgg19 > ban_vgg19.out 2>&1 &
nohup python train_ban.py --gpu 7 --model resnet32 > ban_resnet32.out 2>&1 &
nohup python train_ban.py --gpu 6 --model wide_resnet20_8 > ban_wide_resnet20_8.out 2>&1 &

#4.CSKD(finished)
nohup python train_CSKD.py --gpu 7 --model vgg19 > CSKD_vgg19.out 2>&1 &
nohup python train_CSKD.py --gpu 0 --model resnet32 > CSKD_resnet32.out 2>&1 &
nohup python train_CSKD.py --gpu 1 --model wide_resnet20_8 > CSKD_wide_resnet20_8.out 2>&1 &

#5.SD(finished)
nohup python train_SD.py --gpu 0 --arch vgg19 > SD_vgg19.out 2>&1 &
nohup python train_SD.py --gpu 1 --arch resnet32 > SD_resnet32.out 2>&1 &
nohup python train_SD.py --gpu 3 --arch wide_resnet20_8 >  SD_wide_resnet20_8.out 2>&1 &

#6.Be your own teacher
nohup python train_own_teacher.py --gpu 1 --arch resnet32 > ownteacher_resnet32_kd.out 2>&1 &
nohup python train_own_teacher.py --gpu 0 --arch wide_resnet20_8 > ownteacher_wide_resnet20_8.out 2>&1 &

#7.SD_atten
nohup python train_SD_attention.py --atten 3 --gpu 0 --arch vgg19 --step 3 --warm_up 0 > SD_step3_atten3_vgg19.out 2>&1 &
nohup python train_SD_attention.py --atten 3 --gpu 1 --arch vgg19 --step 5 --warm_up 0 > SD_step5_atten3_vgg19.out 2>&1 &
nohup python train_SD_attention.py --atten 5 --gpu 2 --arch vgg19 --step 3 --warm_up 0 > SD_step3_atten5_vgg19.out 2>&1 &
nohup python train_SD_attention.py --atten 5 --gpu 3 --arch vgg19 --step 5 --warm_up 0 > SD_step5_atten5_vgg19.out 2>&1 &

nohup python train_SD_attention.py --atten 3 --gpu 1 --arch resnet32 --step 3 --warm_up 100 > SD_step3_atten3_resnet32.out 2>&1 &
nohup python train_SD_attention.py --atten 3 --gpu 1 --arch resnet32 --step 5 --warm_up 100 > SD_step5_atten3_resnet32.out 2>&1 &
nohup python train_SD_attention.py --atten 5 --gpu 5 --arch resnet32 --step 3 --warm_up 100 > SD_step3_atten5_resnet32.out 2>&1 &
nohup python train_SD_attention.py --atten 5 --gpu 5 --arch resnet32 --step 5 --warm_up 100 > SD_step5_atten5_resnet32.out 2>&1 &

nohup python train_SD_attention.py --atten 3 --gpu 4 --arch wide_resnet20_8 --step 3 --warm_up 100 > SD_step3_atten3_wide_resnet20_8.out 2>&1 &
nohup python train_SD_attention.py --atten 3 --gpu 4 --arch wide_resnet20_8 --step 5 --warm_up 100 > SD_step5_atten3_wide_resnet20_8.out 2>&1 &
nohup python train_SD_attention.py --atten 5 --gpu 0 --arch wide_resnet20_8 --step 3 --warm_up 100 > SD_step3_atten5_wide_resnet20_8.out 2>&1 &
nohup python train_SD_attention.py --atten 5 --gpu 1 --arch wide_resnet20_8 --step 5 --warm_up 100 > SD_step5_atten5_wide_resnet20_8.out 2>&1 &

#8.New Method
nohup python train_New.py --gpu 0 --arch vgg19 --outdir save_New_V4_atten1 --factor 8 --atten 3 > New_vgg19_atten3.out 2>&1 &
nohup python train_New.py --gpu 1 --arch resnet32 --outdir save_New_V4_atten1 --factor 8 --atten 3 > New_resnet32_atten3.out 2>&1 &
nohup python train_New.py --gpu 2,3 --arch wide_resnet20_8 --outdir save_New_V4_atten1 --factor 8 --atten 3 > New_wide_resnet20_8_atten3.out 2>&1 &

nohup python train_New.py --gpu 1 --arch vgg19 --outdir save_New_V4_atten1 --factor 8 --atten 1 > New_vgg19_atten3.out 2>&1 &
nohup python train_New.py --gpu 0 --arch resnet32 --outdir save_New_V4_atten1 --factor 8 --atten 1 > New_resnet32_atten3.out 2>&1 &
nohup python train_New.py --gpu 4,5 --arch wide_resnet20_8 --outdir save_New_V4_atten1 --factor 8 --atten 1 > New_wide_resnet20_8_atten3.out 2>&1 &

nohup python train_New.py --gpu 4 --tea_avg --arch vgg19 --outdir save_New_V4_avg1 --atten 3 > New_vgg19_avg.out 2>&1 &
nohup python train_New.py --gpu 5 --tea_avg --arch resnet32 --outdir save_New_V4_avg1 --atten 3 > New_resnet32_avg.out 2>&1 &
nohup python train_New.py --gpu 6,7 --tea_avg --arch wide_resnet20_8 --outdir save_New_V4_avg1 --atten 3 > New_wide_resnet20_8_avg.out 2>&1 &

#9.New Method2
nohup python train_New2.py --gpu 0 --arch vgg19 > New2_vgg19.out 2>&1 &
nohup python train_New2.py --gpu 2 --arch resnet32 --outdir save_New2_KD --sd_KD > New2_resnet32.out 2>&1 &
nohup python train_New2.py --gpu 1 --arch wide_resnet20_8 > New2_wide_resnet20_8.out 2>&1 &

#10.change_LWR  用来检测是不是因为每个minibatch更新软标签导致效果的提升（修改为每个epoch更新标签）
nohup python change_LWR.py --gpu 0 --model resnet32 --outdir save_change_LWR3 > change_LWR_resnet32.out 2>&1 &
nohup python change_LWR.py --gpu 1 --model vgg19 --outdir save_change_LWR3 > change_LWR_vgg19.out 2>&1 &
nohup python change_LWR.py --gpu 3 --model wide_resnet20_8 --outdir save_change_LWR3 > change_LWR_wide_resnet20_8.out 2>&1 &
