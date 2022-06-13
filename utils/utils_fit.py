import torch
from tqdm import tqdm
import numpy as np

from utils.utils import get_lr
from IAN import fusion
        
def fit_one_epoch(rgb_model_train, rgb_model, lwir_model_train, lwir_model, IAN_train, IAN, yolo_loss, ian_loss, loss_history, 
                    rgb_optimizer, lwir_optimizer, ian_optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, device):
    loss        = 0
    val_loss    = 0
    train_loss_ian = 0
    val_loss_ian   = 0

    rgb_model_train.train()
    lwir_model_train.train()
    IAN_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, (visible_batch, lwir_batch, ian_batch) in enumerate(gen):
            with torch.autograd.set_detect_anomaly(True):
                if iteration >= epoch_step:
                    break
                
                visible_images, targets = visible_batch[0], visible_batch[1]
                lwir_images, targets = lwir_batch[0], lwir_batch[1]
                ian_images, labels = ian_batch[0], ian_batch[1]
                with torch.no_grad():
                    if cuda:
                        visible_images  = torch.from_numpy(visible_images).type(torch.FloatTensor).cuda()
                        lwir_images  = torch.from_numpy(lwir_images).type(torch.FloatTensor).cuda()
                        ian_images  = torch.from_numpy(ian_images).type(torch.FloatTensor)
                        targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                        labels = torch.tensor([torch.from_numpy(ann).type(torch.FloatTensor) for ann in labels])
                    else:
                        visible_images  = torch.from_numpy(visible_images).type(torch.FloatTensor)
                        lwir_images  = torch.from_numpy(lwir_images).type(torch.FloatTensor)
                        ian_images  = torch.from_numpy(ian_images).type(torch.FloatTensor)
                        targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                        labels = torch.tensor([torch.from_numpy(ann).type(torch.FloatTensor) for ann in labels])
                #----------------------#
                #   清零梯度
                #----------------------#
                rgb_optimizer.zero_grad()
                #----------------------#
                #   forward
                #----------------------#
                visible_outputs         = rgb_model_train(visible_images)
                #----------------------#
                #   清零梯度
                #----------------------#
                lwir_optimizer.zero_grad()
                #----------------------#
                #   forward
                #----------------------#
                lwir_outputs         = lwir_model_train(lwir_images)

                ian_optimizer.zero_grad()
                ian = IAN.forward(visible_images.to(device))
                loss_ian = ian_loss(ian[0], labels.to(device, dtype=torch.int64))
                train_loss_ian = train_loss_ian + loss_ian.item()
                loss_ian.backward(retain_graph=True)
                
                outputs = fusion(visible_outputs, lwir_outputs, ian[-1])

                # output_rgb = (torch.zeros([8, 24, 13, 13], dtype=torch.float).cuda(), torch.zeros([8, 24, 26, 26], dtype=torch.float).cuda(), torch.zeros([8, 24, 52, 52], dtype=torch.float).cuda())
                # output_lwir = (torch.zeros([8, 24, 13, 13], dtype=torch.float).cuda(), torch.zeros([8, 24, 26, 26], dtype=torch.float).cuda(), torch.zeros([8, 24, 52, 52], dtype=torch.float).cuda())
                # for i, x in enumerate(visible_outputs):
                #     for j, y in enumerate(x):
                #         output_rgb[i][j] = torch.mul(y, ian[-1][j])
                # for i, x in enumerate(lwir_outputs):
                #     for j, y in enumerate(x):
                #         output_lwir[i][j] = torch.mul(y, 1-ian[-1][j])

                # outputs = (torch.add(output_rgb[0], output_lwir[0]), torch.add(output_rgb[1], output_lwir[1]), torch.add(output_rgb[2], output_lwir[2]))
            
                loss_value_all  = 0
                num_pos_all     = 0
                #----------------------#
                #   計算loss
                #----------------------#
                for l in range(len(outputs)):
                    loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                    loss_value_all  = loss_value_all + loss_item
                    num_pos_all     = num_pos_all + num_pos
                loss_value = loss_value_all / num_pos_all
                
                #----------------------#
                #   backward
                #----------------------#
                loss_value.backward()
                ian_optimizer.step()
                rgb_optimizer.step()
                lwir_optimizer.step()

                loss = loss + loss_value.item()
                
                pbar.set_postfix(**{'yolo_loss'  : loss / (iteration + 1), 
                                    'rgb_yolo_lr'    : get_lr(rgb_optimizer),
                                    'lwir_yolo_lr'    : get_lr(rgb_optimizer),
                                    'ian_loss'   : train_loss_ian / (iteration + 1),
                                    'ian_lr'     : get_lr(ian_optimizer)})
                pbar.update(1)


    print('Finish Train')

    rgb_model_train.eval()
    lwir_model_train.eval()
    IAN_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, (visible_batch, lwir_batch, ian_batch) in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break

            visible_images, targets = visible_batch[0], visible_batch[1]
            lwir_images, targets = lwir_batch[0], lwir_batch[1]
            ian_images, labels = ian_batch[0], ian_batch[1]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                if cuda:
                    visible_images  = torch.from_numpy(visible_images).type(torch.FloatTensor).cuda()
                    lwir_images  = torch.from_numpy(lwir_images).type(torch.FloatTensor).cuda()
                    ian_images  = torch.from_numpy(ian_images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                    labels = torch.tensor([torch.from_numpy(ann).type(torch.FloatTensor) for ann in labels])
                else:
                    visible_images  = torch.from_numpy(visible_images).type(torch.FloatTensor)
                    lwir_images  = torch.from_numpy(lwir_images).type(torch.FloatTensor)
                    ian_images  = torch.from_numpy(ian_images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                    labels = torch.tensor([torch.from_numpy(ann).type(torch.FloatTensor) for ann in labels])
                #----------------------#
                #   清零梯度
                #----------------------#
                rgb_optimizer.zero_grad()
                #----------------------#
                #   forward
                #----------------------#
                visible_outputs         = rgb_model_train(visible_images)
                #----------------------#
                #   清零梯度
                #----------------------#
                lwir_optimizer.zero_grad()
                #----------------------#
                #   forward
                #----------------------#
                lwir_outputs         = lwir_model_train(lwir_images)

                ian_optimizer.zero_grad()
                ian = IAN.forward(visible_images.to(device))
                loss_ian = ian_loss(ian[0], labels.to(device, dtype=torch.int64))
                val_loss_ian += loss_ian

                outputs = fusion(visible_outputs, lwir_outputs, ian[-1])

                loss_value_all  = 0
                num_pos_all     = 0
                #----------------------#
                #   計算loss
                #----------------------#
                for l in range(len(outputs)):
                    loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                loss_value  = loss_value_all / num_pos_all

            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss'      : val_loss / (iteration + 1),
                                'ian_val_loss'  :val_loss_ian / (iteration + 1)})
            pbar.update(1)
            
    print('Finish Validation')
    
    loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    torch.save({"rgb":rgb_model.state_dict(), "lwir":lwir_model.state_dict(), "ian":IAN.state_dict()},'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))

