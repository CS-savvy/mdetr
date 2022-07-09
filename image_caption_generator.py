from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
import torch
from PIL import Image


class captionGenerator:

  def __init__(self, max_length=16, num_beam=4) -> None:
    self.feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = self.model.to(self.device)
    self.max_length = max_length
    self.num_beams = num_beam
    self.gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams}

  def predict_caption(self, image_paths):
    images = []
    for image_path in image_paths:
      i_image = Image.open(image_path)
      if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

      images.append(i_image)

    pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(self.device)

    output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

    preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


if __name__ == "__main__":
  
  capGen = captionGenerator()

  image_files = ['/home/mukul/Downloads/image_4.jpeg', '/home/mukul/Downloads/image_5.jpeg',
                 '/home/mukul/Downloads/image_6.jpeg', '/home/mukul/Downloads/image_7.jpeg']
  
  output = capGen.predict_caption(image_files)                       
  print(output)
