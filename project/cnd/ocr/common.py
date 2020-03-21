import torch

alphabet = 'ABEKMHOPCTYX' + '0123456789' + '-'

model_parameters = {
    'image_height': 32,
    'number_input_channels': 1,
    'number_class_symbols': len(alphabet),
    'rnn_size': 64
}


def preds_converter(converter, logits, len_images):
    preds_size = torch.IntTensor([logits.size(0)] * len_images)
    _, preds = logits.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(preds, preds_size, raw=False)
    return sim_preds, preds_size