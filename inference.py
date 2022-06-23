from matplotlib.pyplot import draw
from matplotlib.transforms import Transform
import yaml
from pathlib import Path
from models import build_model
from munch import DefaultMunch
from transformers import RobertaTokenizerFast
from PIL import Image, ImageDraw
from  datasets.coco import make_coco_transforms
from util.misc import NestedTensor
import torch
from util.infer_utils import infer, plot_bbox
from models.postprocessors import PostProcess
import torch.nn.functional as F

args = {
    'resume': '/media/mukul/Storage/Projects/MDetr/data/models/pretrained_resnet101_checkpoint.pth',
    'model_config_file': '/media/mukul/Storage/Projects/MDetr/mdetr/configs/model_conf.yaml' 
    }

if 'model_config_file' in args:
    with open(args['model_config_file'], 'r') as f:
        config = yaml.safe_load(f)
    args.update(config)


dataset = [{
    'image_file': '/home/mukul/Downloads/WhatsApp Image 2022-06-14 at 6.01.29 PM (1).jpeg',
    'queries': [
        "A ant is bugging the bug.",
        "A ant is bugging the bug.",]
    },
    {
    'image_file': '/home/mukul/Downloads/WhatsApp Image 2022-06-14 at 6.01.29 PM (1).jpeg',
    'queries': [
        "A ant is bugging the bug.",
        "A ant is bugging the bug.",
    ]
    }
]

out_dir = Path(r"/media/mukul/Storage/Projects/MDetr/data/flickr/output")

transform = make_coco_transforms('val', cautious=True)

args = DefaultMunch.fromDict(args)
device = torch.device(args.device)

model, criterion, _, _, _ = build_model(args)

checkpoint = torch.load(args.resume, map_location="cpu")
model.load_state_dict(checkpoint["model"])

print("Model loaded")

model.to(device)

bbox_postprocessor = PostProcess()

box_threshold = 0.9
token_threshold = 0.1

for data in dataset:
    image_pil = Image.open(data['image_file'])
    w, h = image_pil.size
    image_pil = image_pil.resize((w*2, h*2))
    w, h = image_pil.size
    out_path = out_dir / data['image_file'].split("/")[-1]
    drawer = ImageDraw.Draw(image_pil)
    captions = data['queries'][:2]
    target_size = torch.as_tensor([(int(h), int(w)), (int(h), int(w))]).to(device)
    image, _ = transform(image_pil, {})
    batch_images = NestedTensor.from_tensor_list([image, image], do_round=False)
    batch_images = batch_images.to(device)
    output = infer(model, batch_images, captions)
    bbox_details = bbox_postprocessor(output, target_size)
    pred_prod = F.softmax(output['pred_logits'], -1)
    for bbox in bbox_details:

        scores = bbox['scores']
        labels = bbox['labels']
        boxes = bbox['boxes']
        mask = torch.ones(scores.size()[0]).to(device)
        indices = torch.nonzero((scores >= box_threshold)*mask)
        indices = [int(k) for k in indices.detach().cpu().numpy()]
        
        phrases = {}
        for ind in indices:
            prob = pred_prod[0][ind]
            tok_mask = torch.ones(prob.size()[0]).to(device)
            tok_indices = torch.nonzero((prob >=token_threshold)*tok_mask)
            tok_indices = [int(k) for k in tok_indices.detach().cpu().numpy()]
            beg = min(tok_indices)
            end = max(tok_indices)
            span_start = output['tokenized'].token_to_chars(0, beg)
            span_end = output['tokenized'].token_to_chars(0, end)
            
            phrase = captions[0][span_start.start: span_end.end]
            if phrase in phrases:
                phrases[phrase].append((ind, float(scores[ind])))
            else:
                phrases[phrase] = [(ind, float(scores[ind]))]


        plot_bbox(drawer, boxes, scores, phrases, captions[0])
        image_pil.save(out_path)

        break

# dataset
# tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type, local_files_only=True)