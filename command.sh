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
nohup python train_New.py --gpu 3 --arch vgg19 --outdir save_New_2 --atten 1 > New_vgg19.out 2>&1 &
nohup python train_New.py --gpu 4 --arch resnet32 --outdir save_New_2 --atten 1 > New_resnet32.out 2>&1 &
nohup python train_New.py --gpu 5 --arch wide_resnet20_8 --outdir save_New_2 --atten 1 > New_wide_resnet20_8.out 2>&1 &

nohup python train_New.py --gpu 5 --arch vgg19 --outdir save_New_2 > New_vgg19.out 2>&1 &
nohup python train_New.py --gpu 5 --arch resnet32 --outdir save_New_2 > New_resnet32.out 2>&1 &
nohup python train_New.py --gpu 5 --arch wide_resnet20_8 --outdir save_New_2 > New_wide_resnet20_8.out 2>&1 &

nohup python train_New.py --gpu 6 --arch vgg19 --outdir save_New_3 > New_vgg19.out 2>&1 &
nohup python train_New.py --gpu 6 --arch resnet32 --outdir save_New_3 > New_resnet32.out 2>&1 &
nohup python train_New.py --gpu 6 --arch wide_resnet20_8 --outdir save_New_3 > New_wide_resnet20_8.out 2>&1 &

#8.New Method2
nohup python train_New2.py --gpu 0 --arch vgg19 > New2_vgg19.out 2>&1 &
nohup python train_New2.py --gpu 2 --arch resnet32 --outdir save_New2_KD --sd_KD > New2_resnet32.out 2>&1 &
nohup python train_New2.py --gpu 1 --arch wide_resnet20_8 > New2_wide_resnet20_8.out 2>&1 &

#9.change_LWR  用来检测是不是因为每个minibatch更新软标签导致效果的提升（修改为每个epoch更新标签）
nohup python change_LWR.py --gpu 6 --model resnet32 --outdir save_fun > change_LWR_resnet32.out 2>&1 &
nohup python change_LWR.py --gpu 1 --model vgg19 --outdir save_change_LWR_test > change_LWR_vgg19.out 2>&1 &
nohup python change_LWR.py --gpu 4 --model wide_resnet20_8 --outdir save_change_LWR_test > change_LWR_wide_resnet20_8.out 2>&1 &

#9.5.change_LWR2 用来检测是不是因为每个k周期将kdloss归零导致效果的提升（除掉归0的那个epoch） 正常
nohup python change_LWR2.py --gpu 3 --model resnet32 > change_LWR2_resnet32.out 2>&1 &
nohup python change_LWR2.py --gpu 4 --model vgg19 > change_LWR2_vgg19.out 2>&1 &
nohup python change_LWR2.py --gpu 5 --model wide_resnet20_8 > change_LWR2_wide_resnet20_8.out 2>&1 &

#9.6.change_LWR3 用来检测loss前的alpha权重系数的作用效果 正常
nohup python change_LWR3.py --gpu 3 --model resnet32 > change_LWR3_resnet32.out 2>&1 &
nohup python change_LWR3.py --gpu 0 --model vgg19 > change_LWR3_vgg19.out 2>&1 &
nohup python change_LWR3.py --gpu 5 --model wide_resnet20_8 > change_LWR3_wide_resnet20_8.out 2>&1 &

#10.see_LWR
nohup python see_LWR.py --gpu 0 --model resnet32 > see_LWR_resnet32.out 2>&1 &
nohup python see_LWR.py --gpu 1 --model vgg19 > see_LWR_vgg19.out 2>&1 &
nohup python see_LWR.py --gpu 2 --model wide_resnet20_8 > see_LWR_wide_resnet20_8.out 2>&1 &


#save_SD_atten_V6现在运行的是loss函数当中没有对两种loss加权重的版本
#save_New_V2运行的是带有余弦退火的版本

#nohup python Test.py --gpu 3 --model vgg19 > test_vgg19.out 2>&1 &
#nohup python Test.py --gpu 1 --model resnet32 > test_resnet32.out 2>&1 &
#nohup python Test.py --gpu 2 --model wide_resnet20_8 > test_wide_resnet20_8.out 2>&1 &

#nohup python test_SD_attention.py --gpu 4 --arch resnet32 > test_atten_resnet32.out 2>&1 &
#nohup python test_SD_attention.py --gpu 5 --arch vgg19 > test_atten_vgg19.out 2>&1 &
#nohup python test_SD_attention.py --gpu 6 --arch wide_resnet20_8 > test_atten_wide_resnet20_8.out 2>&1 &

#scp -P 10103 -r wbh@10.76.2.231:/home/wbh/Self_KD /home/wbh

nohup python change_LWR.py --gpu 0 --model resnet32 > change_LWR_resnet32.out 2>&1 &  # 将LWR改成一个epoch更新label
# save_New_V2代表权重为1 save_New_V1权重衰减
# TODO:打印LWR的kd_Loss和label_loss

nohup python train_New_Test.py --gpu 3 --arch vgg19 --outdir save_New_Test --sd_KD --atten 1 > New_vgg19_Test.out 2>&1 &
nohup python train_New_Test.py --gpu 4 --arch resnet32 --outdir save_New_Test --sd_KD --atten 1 > New_resnet32_Test.out 2>&1 &
nohup python train_New_Test.py --gpu 5 --arch wide_resnet20_8 --outdir save_New_Test --sd_KD --atten 1 > New_wide_resnet20_8_Test.out 2>&1 &


#TODO：显存膨胀问题！ deepcopy