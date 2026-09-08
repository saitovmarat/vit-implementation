from torchvision import transforms


def get_train_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def get_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

