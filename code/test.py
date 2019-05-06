from train_val import *
from Levenshtein import distance


def test(model, test_loader):
    global answer_path

    model.eval()

    idx = 0

    answer_dict = {}
    
    with torch.no_grad():
        for inputs in test_loader:
            print("================")
            print("term:", idx)
            predictions = model.predict(inputs, 'test')

            for pred in predictions:
                pred_str = toSentence(pred)
                pred_str = pred_str[1:-1]
            
                # print("pred sentence:", pred_str)

                answer_dict[idx] = pred_str

                idx += 1

        np.savez(answer_path, answer_dict)
    return


if __name__ == '__main__':

    test_loader = get_loader('test', batch_size=1) # use the test dataset to perform beam search

    result_path = './../result/model_exp7_19.t7'
    model, _ = load_ckpt(result_path, 'test')
    model = model.to(DEVICE)

    test(model, test_loader)
