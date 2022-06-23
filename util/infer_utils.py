from turtle import width
import torch


@torch.no_grad()
def infer(model, samples, captions):

    model.eval()
    memory_cache = model(samples, captions, encode_and_save=True)
    outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)
    
    return outputs


def plot_bbox(image_drawer, bbox, scores, phrases, caption):

    for phrase, val in phrases.items():
        
        val = sorted(val, key=lambda x: x, reverse=True)
        ind = val[0][0]
        x1,y1,x2,y2 = bbox[ind]
        # score = scores[ind]
        image_drawer.rectangle([int(x1), int(y1), int(x2), int(y2)], width=2, outline=(255, 0, 0))
        image_drawer.text([int(x1), int(y1)], f"{phrase}")
        image_drawer.text([10, 10], f"{caption}")
