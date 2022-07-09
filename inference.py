import yaml
from pathlib import Path
from models import build_model
from munch import DefaultMunch
from PIL import Image, ImageDraw
from  datasets.coco import make_coco_transforms
from util.misc import NestedTensor
import torch
from util.infer_utils import infer, plot_bbox
from models.postprocessors import PostProcess
import torch.nn.functional as F


class coRegionResolution:

    def __init__(self, model_weights, model_config) -> None:      
        self.args = {
            'resume': model_weights,
            'model_config_file': model_config
            }
        
        with open(self.args['model_config_file'], 'r') as f:
            self.config = yaml.safe_load(f)
        self.args.update(self.config)
        self.args = DefaultMunch.fromDict(self.args)
        self.device = torch.device(self.args.device)
        self.model, self.criterion, _, _, _ = build_model(self.args)
        self.transform = make_coco_transforms('val', cautious=True)
        self.checkpoint = torch.load(self.args.resume, map_location="cpu")
        self.model.load_state_dict(self.checkpoint["model"])
        self.model.to(self.device)
        self.bbox_postprocessor = PostProcess()
        self.box_threshold = 0.5
        self.token_threshold = 0.3

    def preprocess(self, image_paths):
        images = []
        target_sizes = []
        
        for image_file in image_paths:
            image_pil = Image.open(image_file)
            w, h = image_pil.size
            image_pil = image_pil.resize((w*2, h*2))
            w, h = image_pil.size
            image, _ = self.transform(image_pil, {})
            images.append(image)
            target_sizes.append((int(h), int(w)))
        batch_images = NestedTensor.from_tensor_list(images, do_round=False)
        target_size = torch.as_tensor(target_sizes).to(self.device)

        return batch_images, target_size

    def plot_results(self, images, predictions, captions, out_dir, scale = 2):
        assert len(images) == len(predictions)
        for cap, img_path in zip(captions, images):
            out_path = out_dir / f"{img_path.stem}.jpg"
            pred = predictions[img_path.stem]
            pil_image = Image.open(img_path)
            w, h = pil_image.size
            pil_image = pil_image.resize((w*2, h*2))
            w, h = pil_image.size
            drawer = ImageDraw.Draw(pil_image)

            for phrase, (score, box) in pred.items():
                drawer.rectangle(box, width=2, outline=(255, 0, 0))
                drawer.text(box[:2], f"{phrase}-{score:2f}")
                    
            drawer.text([10, 10], f"{cap}")
            pil_image.save(out_path)
    
    def predict(self, image_files, captions):

        assert len(image_files) == len(captions), f"length of images -{len(image_files)} and captions - {captions} are different."
        
        batch_images, target_size = self.preprocess(image_files) 

        batch_images = batch_images.to(self.device)
        target_size = target_size.to(self.device)

        output = infer(self.model, batch_images, captions)
        bbox_details = self.bbox_postprocessor(output, target_size)
        pred_prod = F.softmax(output['pred_logits'], -1)
        
        results = {}
        for i, bbox in enumerate(bbox_details):
            scores = bbox['scores']
            boxes = bbox['boxes']
            # labels = bbox['labels']
            mask = torch.ones(scores.size()[0]).to(self.device)
            indices = torch.nonzero((scores >= self.box_threshold)*mask)
            indices = [int(k) for k in indices.detach().cpu().numpy()]
            
            phrases = {}
            for ind in indices:
                x1,y1,x2,y2 = boxes[ind]
                bbox_coord = int(x1), int(y1), int(x2), int(y2)
                prob = pred_prod[i][ind]
                tok_mask = torch.ones(prob.size()[0]).to(self.device)
                tok_indices = torch.nonzero((prob >=self.token_threshold)*tok_mask)
                tok_indices = [int(k) for k in tok_indices.detach().cpu().numpy()]
                if 255 in tok_indices:
                    tok_indices.remove(255)

                if not tok_indices:
                    continue

                beg = min(tok_indices)
                end = max(tok_indices)
                span_start = output['tokenized'].token_to_chars(i, beg)
                span_end = output['tokenized'].token_to_chars(i, end)
                
                phrase = captions[i][span_start.start: span_end.end]
                if phrase in phrases:
                    current_score = phrases[phrase][0]
                    if float(scores[ind]) > current_score:
                        phrases[phrase] = (float(scores[ind]), bbox_coord)
                else:
                    phrases[phrase] = (float(scores[ind]), bbox_coord)

            results[image_files[i].stem] = phrases

        return results


if __name__ == "__main__":
    
    model_weights = Path(r'/media/mukul/Storage/Projects/MDetr/data/models/pretrained_resnet101_checkpoint.pth')
    model_config = Path(r'/media/mukul/Storage/Projects/MDetr/mdetr/configs/model_conf.yaml')
    output_path = Path(r"/media/mukul/Storage/Projects/MDetr/data/flickr/output")

    crr = coRegionResolution(model_weights, model_config)

    images = [Path(r'/home/mukul/Downloads/image_5.jpeg')]

    captions = ['A Ant and a Bug on leaf.']

    out = crr.predict(images, captions)
    crr.plot_results(images, out, captions, output_path)



# Path(r'/home/mukul/Downloads/image_5.jpeg')
# Path(r'/home/mukul/Downloads/image_4.jpeg')
# Path(r'/home/mukul/Downloads/image_2.jpeg')
# Path(r'/home/mukul/Downloads/image_8.jpeg')

# 'A Ant and a Bug on leaf.'
# 'A child holding a banana'
# 'A basket filled with fruits.'
# 'A coin and an ant laying on a surface.'