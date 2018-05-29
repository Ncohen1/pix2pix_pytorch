python3 train.py --dataroot ./datasets/ctUs_Data_cropped --name ctUs_pix2pix_withDropout --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --input_nc 1 --output_nc 1 --niter 1 --niter_decay 1 --continue_train  


#python3 train.py --dataroot ./datasets/ctUs_Data_cropped --name ctUs_pix2pix_noDropout --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --input_nc 1 --output_nc 1 --niter 25 --niter_decay 25 --no_dropout 
#24_4 - first try - for pix2pix we used only our own flags, we also needed dropout!
#python3 train.py --dataroot ./datasets/ctUs_Data_cropped --name ctUsTrain_changesOptions_pix2pix --model pix2pix --pool_size 0 --no_dropout --input_nc 1 --output_nc 1 --gpu_ids 0,2 --dataset_mode aligned --identity 0.1 --niter 25 -#-niter_decay 25
#python3 train.py --dataroot ./datasets/ctUs_Data_cropped --name ctUsTrain_changeOptions_cyclegan --model cycle_gan --pool_size 50 --no_dropout --input_nc 1 --output_nc 1 --gpu_ids 0,2 --dataset_mode aligned --identity 0.1 --niter 25 --niter_decay 25
