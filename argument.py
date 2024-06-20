import argparse

def linear_eval_argument():
    parser = argparse.ArgumentParser(description='linear evaluation test')
    parser.add_argument('--rLE', action='store_true', help='linear evaulation with adversarial examples')
    parser.add_argument('--rFT', action='store_true', help='adversarial full finetuning with adversarial examples')
    parser.add_argument('--adv_train', default='AT', type=str, help='train_method standard AT/ TRADES')
    parser.add_argument('--max_epochs', default=100, type=int, help='max epochs')
    
    parser.add_argument('--target', default='cifar10', type=str, help='dataset')
    
    parser.add_argument('--batch_size_per_gpu', default=1024, type=int, help='batch size per gpu')
    parser.add_argument('--schedule', default='step', type=str, help='scheduler type (step/ cosine)')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='warm up epoch')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    
    parser.add_argument('--num_worker', default=4, type=int, help='number of workers')
    parser.add_argument('--device', default='cuda', type=str, help='gpu: cuda/ cpu: cpu')
    
    parser.add_argument('--ckpt_path', default='', type=str, help='loading_path')
    parser.add_argument('--loading_epoch', default='', type=str, help='loading file')

    parser.add_argument('--resume', default=None, type=str, help='resuming the checkpoint')
    parser.add_argument('--fast_eval', action='store_true', help='resuming the checkpoint')
    
    parser.add_argument('--attack_mode', default='supervised', type=str, help='attack mode (selfsup/ supervised)')
    
    parser.add_argument('--epsilon', default=8, type=int, help='attack epsilon size (args.epsilon/255.)')
    parser.add_argument('--alpha', default=2, type=int, help='attack step size (args.alpha/255.)')
    parser.add_argument('--n_iters', default=10, type=int, help='number of attack iteration')
    parser.add_argument('--distance', default='Linf', type=str, help='attack distance (Linf/L2/L1)')
    parser.add_argument('--adv_method', default='pgd', type=str, help='attack method (fgsm/pgd/autoattack)')
    
    parser.add_argument('--fname', default='', type=str, help='additional file name')
    parser.add_argument('--image_size', default=32, type=int)
    
    args = parser.parse_args()
    ckpt_parse = args.ckpt_path.split('/')
    for i in range(len(ckpt_parse)):
        if len(ckpt_parse[i])>20:
            index = i
    find_index = ckpt_parse[index].split('_')
    backbone_index = 0
    target_index = 0
    for i in range(len(find_index)):
        if find_index[i][0]=='M':
            backbone_index = i
        if find_index[i][0]=='D':
            target_index = i
    args.backbone = ckpt_parse[index].split('_')[backbone_index][1:]
    loading_target = ckpt_parse[index].split('_')[target_index][1:]
    if loading_target == args.target:
        print('linear evaluation')
        ans = input("Do you want to linear evaluate the representation? [y/N]? ")
        if not ans in ['y', 'Y']:
            exit(1)
    else:
        print('transfer evaluation')
        ans = input("Do you want to transfer evaluate the representation? [y/N]? ")
        if not ans in ['y', 'Y']:
            exit(1)
        
    print(args.backbone, args.target)
    if args.target == 'cifar5' or args.target == 'cifar10' or args.target == 'cifar100':
        args.image_size = 32
    elif args.target == 'stl10':
        args.image_size = args.image_size
    else:
        args.image_size = 224
    args.save_path = ckpt_parse[index]
    args.ckpt_path = args.ckpt_path 
    args.fname += args.adv_train
    return args

def adv_test_argument():
    parser = argparse.ArgumentParser(description='linear evaluation test')
    
    parser.add_argument('--num_worker', default=4, type=int, help='number of workers')
    parser.add_argument('--device', default='cuda', type=str, help='gpu: cuda/ cpu: cpu')
    
    parser.add_argument('--ckpt_path', default='', type=str, help='loading_path')
    parser.add_argument('--loading_epoch', default='', type=str, help='loading file')

    parser.add_argument('--resume', default=None, type=str, help='resuming the checkpoint')
    
    parser.add_argument('--attack_mode', default='supervised', type=str, help='attack mode (selfsup/ supervised)')
    
    parser.add_argument('--epsilon', default=8, type=float, help='attack epsilon size (args.epsilon/255.)')
    parser.add_argument('--alpha', default=0.8, type=float, help='attack step size (args.alpha/255.)')
    parser.add_argument('--n_iters', default=20, type=int, help='number of attack iteration')
    parser.add_argument('--distance', default='Linf', type=str, help='attack distance (Linf/L2/L1)')
    parser.add_argument('--adv_method', default='pgd', type=str, help='attack method (fgsm/pgd/autoattack)')
    
    parser.add_argument('--fname', default='', type=str, help='additional file name')
    
    args = parser.parse_args()
    ckpt_parse = args.ckpt_path.split('/')
    for i in range(len(ckpt_parse)):
        if len(ckpt_parse[i])>0:
            index = i
    find_index = ckpt_parse[index].split('_')
    backbone_index = 0
    target_index = 0
    for i in range(len(find_index)):
        if find_index[i][0]=='M':
            backbone_index = i
        if find_index[i][0]=='D':
            target_index = i
    args.backbone = ckpt_parse[index].split('_')[backbone_index][1:]
    args.target = ckpt_parse[index].split('_')[target_index][1:]
    print(args.backbone, args.target)
    args.save_path = ckpt_parse[index]
    if args.target == 'cifar5' or args.target == 'cifar10' or args.target == 'cifar100':
        args.image_size = 32
    elif args.target == 'stl10':
        args.image_size = 96
    else:
        args.image_size = 224
    args.ckpt_path =  args.ckpt_path +'/'
    return args

def pretrain_argument():
    parser = argparse.ArgumentParser(description='TARO Experiment')
    parser.add_argument('--data_path', default='./', type=str, help='save path')
    
    parser.add_argument('--alpha1', default=1.0, type=float, help='learning rate')
    parser.add_argument('--alpha2', default=1.0, type=float, help='learning rate')
    
    parser.add_argument('--backbone', default='resnet18', type=str, help='backbone model')
    parser.add_argument('--target', default='cifar10', type=str, help='dataset')

    parser.add_argument('--max_epochs', default=800, type=int, help='max epochs')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='warmup epochs (schedule)')
    
    parser.add_argument('--batch_size_per_gpu', default=512, type=int, help='batch size per gpu')
    
    parser.add_argument('--learning_rate', default=3e-2, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    
    parser.add_argument('--image_size', default=32, type=int, help='cifar:32/ imagenet 224')
    parser.add_argument('--num_worker', default=4, type=int, help='number of workers')
    parser.add_argument('--device', default='cuda', type=str, help='gpu: cuda/ cpu: cpu')
    
    parser.add_argument('--ckpt_path', default='./ckpt_path/', type=str, help='save path')
    parser.add_argument('--resume', default=None, type=str, help='resuming the checkpoint')
    
    parser.add_argument('--attack_mode', default='selfsup', type=str, help='attack mode (selfsup/ supervised)')
    
    parser.add_argument('--epsilon', default=8, type=int, help='attack epsilon size (args.epsilon/255.)')
    parser.add_argument('--alpha', default=2, type=int, help='attack step size (args.alpha/255.)')
    parser.add_argument('--n_iters', default=10, type=int, help='number of attack iteration')
    parser.add_argument('--distance', default='Linf', type=str, help='attack distance (Linf/L2/L1)')
    parser.add_argument('--adv_method', default='pgd', type=str, help='attack method (fgsm/pgd/autoattack)')
    parser.add_argument('--attack_loss', default='sim_loss', type=str, help='attack loss for robust self supervised')
    
    parser.add_argument('--w2', default=1.0, type=float, help='weight on adversarial loss')
    
    parser.add_argument('--fname', default='', type=str, help='additional file name')
    args = parser.parse_args()
    
    if args.target == 'cifar10' or args.target == 'cifar100':
        args.image_size = 32
    elif args.target == 'stl10':
        args.image_size = 32
    else:
        args.image_size = 224
    args.ckpt_path = args.ckpt_path + '/TARO'+str(args.alpha1)+'_'+str(args.alpha2)+'_'+str(args.alpha3)+'_pretrain_D'+args.target+'_M'+args.backbone+'_B'+str(args.batch_size_per_gpu)+'_LR'+str(args.learning_rate)\
    +'_WD'+str(args.weight_decay)+'_AT_ep'+str(args.epsilon)+'al'+str(args.alpha)+'iter'+str(args.n_iters)+'_loss_'+args.attack_loss\
    +'_w2_'+str(args.w2)+args.fname+'/'
    return args
