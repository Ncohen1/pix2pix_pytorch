import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
# yuval_naama
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

# yuval_naama
im_str = []
# err1 = {}
# err2 = {}
loss_vals = []
error_vals ={}
#

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # naama_yuval
    loss_avg = {}
    #
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data) # preprocess input
        model.optimize_parameters() #forward + backwards propagation

        # yuval_naama
        # last epoch save all images
        # if epoch == 1:
        if epoch == opt.niter + opt.niter_decay:
            curr_im_str = data['A_paths'][0].split('/')[-1:][0].split('.')[0]
            im_str.append(curr_im_str)
            # errors = model.get_current_errors()
            # err1.update({int(curr_im_str): errors['G_GAN']})
            # err2.update({int(curr_im_str): errors['G_L1']})
            error_vals.update({int(curr_im_str): model.get_current_errors()})
            visualizer.display_current_results_ny(model.get_current_visuals(), epoch, True, i, im_str, error_vals)
        #
        # yuval_naama
        loss_vals.append(model.get_current_errors())

        if total_steps % 10 == 0:
            for i, key in enumerate(loss_vals[0].keys()):
                loss_avg.update({key: np.average([loss_vals[k][key] for k in range(len(loss_vals))])})
            # t = (time.time() - iter_start_time) / opt.batchSize
            # visualizer.print_current_errors(epoch, epoch_iter, loss_avg, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, loss_avg)
            loss_avg = {}
            loss_vals = []
        #
        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        # if total_steps % opt.print_freq == 0:
        #     errors = model.get_current_errors()
        #     t = (time.time() - iter_start_time) / opt.batchSize
        #     visualizer.print_current_errors(epoch, epoch_iter, errors, t)
        #     if opt.display_id > 0:
        #         visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()

# yuval_naama
# plt.figure()
# plt.title('G_GAN')
# plt.plot(err1.keys(), err1.values(), '*-')
# plt.grid()
# plt.savefig('G_GAN.png')
# plt.figure()
# plt.title('G_L1')
# plt.plot(err2.keys(), err2.values(), '*-')
# plt.grid()
# plt.savefig('G_L1.png')
#
# #pd.DataFrame(list(err1.values()),columns=[err1.keys()])
#
# pd.DataFrame(list([err1.values()]),columns=list(err1.keys())).to_csv('Loss_G_GAN.csv',index=False)
# pd.DataFrame(list([err2.values()]),columns=list(err2.keys())).to_csv('Loss_G_L1.csv',index=False)
#
#
# a=pd.DataFrame(list([err1.values()]),columns=list(err1.keys()))
# b=pd.DataFrame(list([err2.values()]),columns=list(err2.keys()))

# pd.DataFrame(a.stack().values,columns=[['loss']]).to_csv('Loss_G_GAN_stack.csv', header=True, index=True)
# pd.DataFrame(b.stack().values,columns=[['loss']]).to_csv('Loss_G_L1_stack.csv', header=True, index=True)

# plt.show()