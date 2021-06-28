import torch
from tqdm import tqdm
from pandas import DataFrame
from torch.utils.data import DataLoader

from model.model import BaseLineRNN
from utils.dataset import RawAudioAmplitudeDataset
from model.config import DATA_PATH, RUN_NAME, SAVE_PATH

if __name__ == '__main__':
    # predict on device
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    # print("Current device", torch.cuda.get_device_name(torch.cuda.current_device()))

    # call model + load pretrained weight
    model = BaseLineRNN().to(device)
    model.load_state_dict(torch.load(SAVE_PATH+RUN_NAME+'.pth'))
    
    # set layers such as dropout and batchnorm in evaluation mode
    model.eval()

    # TODO: transform
    test_set = RawAudioAmplitudeDataset(DATA_PATH, is_train=False)

    # iterate though all the data
    pbar = tqdm(range(len(test_set)))
    prediction = {}

    for idx in pbar:
        audio_id, audio_tensor, sampling_rate = test_set[idx]

        # forward
        pred = model(audio_tensor.unsqueeze(-1))[0]

        print("FORWARD", audio_tensor.unsqueeze(0).shape, "PRED", pred)

        pred = pred.data.cpu().numpy()
        prediction[audio_id] = (audio_id, pred)

    # save results
    df = DataFrame.from_dict(prediction, orient='index', columns=['uuid', 'assessment_result'])

    df.to_csv('results.csv', index=False)



