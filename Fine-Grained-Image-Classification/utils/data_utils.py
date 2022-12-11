from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from .dataset import CUB


def get_loader(args):
    train_transform = transforms.Compose([transforms.Resize((600, 600,), Image.BILINEAR),
                                          transforms.RandomCrop((448, 448)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                         transforms.CenterCrop((448, 448)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    training_data = CUB(root=args.data_root, is_train=True, transform=train_transform)
    testing_data = CUB(root=args.data_root, is_train=False, transform=test_transform)

    train_sampler = RandomSampler(training_data)
    test_sampler = SequentialSampler(testing_data)
    train_loader = DataLoader(training_data,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(testing_data,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testing_data is not None else None
    return train_loader, test_loader
