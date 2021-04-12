##对比方法：
#1.baseline:(finished)

nohup python train_baseline.py --gpu 0 --model vgg19 > baseline_vgg19.out 2>&1 &

nohup python train_baseline.py --gpu 1 --model resnet32 > baseline_resnet32.out 2>&1 &

nohup python train_baseline.py --gpu 2 --model wide_resnet20_8 > baseline_wide_resnet20_8.out 2>&1 &


#2.LWR:(finished)
nohup python train_LWR.py --gpu 1 --model vgg19 > LWR_vgg19.out 2>&1 &

nohup python train_LWR.py --gpu 3 --model resnet32 > LWR_resnet32.out 2>&1 &

nohup python train_LWR.py --gpu 0 --model wide_resnet20_8 > LWR_wide_resnet20_8.out 2>&1 &

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
nohup python train_SD_attention.py --atten 3 --gpu 0 --arch vgg19 --step 3 --warm_up 210 > SD_step3_atten3_vgg19.out 2>&1 &
nohup python train_SD_attention.py --atten 3 --gpu 0 --arch vgg19 --step 5 --warm_up 210 > SD_step5_atten3_vgg19.out 2>&1 &
nohup python train_SD_attention.py --atten 5 --gpu 1 --arch vgg19 --step 3 --warm_up 210 > SD_step3_atten5_vgg19.out 2>&1 &
nohup python train_SD_attention.py --atten 5 --gpu 1 --arch vgg19 --step 5 --warm_up 210 > SD_step5_atten5_vgg19.out 2>&1 &

nohup python train_SD_attention.py --atten 3 --gpu 3 --arch resnet32 --step 3 --warm_up 210 > SD_step3_atten3_resnet32.out 2>&1 &
nohup python train_SD_attention.py --atten 3 --gpu 4 --arch resnet32 --step 5 --warm_up 210 > SD_step5_atten3_resnet32.out 2>&1 &
nohup python train_SD_attention.py --atten 5 --gpu 5 --arch resnet32 --step 3 --warm_up 210 > SD_step3_atten5_resnet32.out 2>&1 &
nohup python train_SD_attention.py --atten 5 --gpu 6 --arch resnet32 --step 5 --warm_up 210 > SD_step5_atten5_resnet32.out 2>&1 &

nohup python train_SD_attention.py --atten 3 --gpu 2 --arch wide_resnet20_8 --step 3 --warm_up 210 > SD_step3_atten3_wide_resnet20_8.out 2>&1 &
nohup python train_SD_attention.py --atten 3 --gpu 2 --arch wide_resnet20_8 --step 5 --warm_up 210 > SD_step5_atten3_wide_resnet20_8.out 2>&1 &
nohup python train_SD_attention.py --atten 5 --gpu 3 --arch wide_resnet20_8 --step 3 --warm_up 210 > SD_step3_atten5_wide_resnet20_8.out 2>&1 &
nohup python train_SD_attention.py --atten 5 --gpu 3 --arch wide_resnet20_8 --step 5 --warm_up 210 > SD_step5_atten5_wide_resnet20_8.out 2>&1 &



# save_SD_atten_V5 添加权重系数
# save_SD_atten_V3 未加loss1和loss2的权重
# save_SD_atten_V2 未加loss1和loss2的权重 未除去KDloss的T方
