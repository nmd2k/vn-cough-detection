import os
import torch
from tqdm import tqdm
from pandas import DataFrame
from torch.utils.data import DataLoader

from model.model import Randomize
from utils.dataset import AICoughDataset
from utils.data_tools import validate_submission
from model.config import DATA_PATH, RUN_NAME, SAVE_PATH

if __name__ == '__main__':
    # predict on device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Current device", torch.cuda.get_device_name(torch.cuda.current_device()))

    # call model + load pretrained weight
    model = Randomize().to(device)
    model.load_state_dict(torch.load(os.path.join(SAVE_PATH,RUN_NAME+'.pth')))
    
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

        pred = pred.detach().cpu().numpy()
        prediction[idx] = (id, pred[0])

    # save results
    df = DataFrame.from_dict(prediction, orient='index', columns=['uuid', 'assessment_result'])

    print(df)

    path = 'results.csv'
    # compression_opts = dict(method='zip', archive_name='results.zip')
    df.to_csv(path, index=False)

    # validation
    validate_submission(os.path.join(DATA_PATH, 'test', 'metadata.csv'), path)



