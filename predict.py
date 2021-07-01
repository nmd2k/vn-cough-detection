import os
import torch
import argparse
from tqdm import tqdm
from pandas import DataFrame
from torch.utils.data import DataLoader

from model.model import Randomize, SimpleCNN, initialize_model
from utils.dataset import AICoughDataset
from utils.data_tools import validate_submission
from model.config import DATA_PATH, RUN_NAME, SAVE_PATH

def parse_args():
    """
    Parse command line arguments
    """
    
    parser = argparse.ArgumentParser(description='COVID-19 Detection through Cough')
    parser.add_argument('--model', type=str, default='resnet18', help='model name (default: resnet18)')
    parser.add_argument('--weight', type=str, default='./model/exp1/weight.pth', help='path to weight')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    # predict on device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Current device", torch.cuda.get_device_name(torch.cuda.current_device()))

    # call model + load pretrained weight
    model = initialize_model(model_name=args.model).to(device)
    model.load_state_dict(torch.load(args.weight))
    
    # set layers such as dropout and batchnorm in evaluation mode
    model.eval()

    # TODO: transform
    test_set = AICoughDataset(DATA_PATH, is_train=False)

    # iterate though all the data
    pbar = tqdm(range(len(test_set)))
    prediction = {}

    for idx in pbar:
        id, input = test_set[idx]

        # forward
        input = input.to(device).unsqueeze(0)
        pred = model(input)

        pred = pred.squeeze().detach().cpu().numpy()
        prediction[idx] = (id, pred)

    # save results
    df = DataFrame.from_dict(prediction, orient='index', columns=['uuid', 'assessment_result'])

    print(df)

    path = 'results.csv'
    # compression_opts = dict(method='zip', archive_name='results.zip')
    df.to_csv(path, index=False)

    # validation
    validate_submission(os.path.join(DATA_PATH, 'test', 'metadata.csv'), path)



