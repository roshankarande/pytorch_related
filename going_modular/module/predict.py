
"""predict the label when fed with an image_path"""

import torchvision
import torch
import torchvision.transforms.v2 as transforms
import argparse
import model as model_repo

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

class_names = ['pizza', 'steak', 'sushi']

def get_args_parser(add_help=True):
    
    parser = argparse.ArgumentParser(description="PyTorch Classification Prediction", add_help=add_help)

    parser.add_argument(
                    "--img",
                    default="../data/04-pizza-dad.jpeg",
                    type=str,
                    help="image path" )

    parser.add_argument(
            "--model_state_path",
            default="../models/tinyvgg_model_v1.pth",
            type=str,
            help="model_state_path")

    return parser

def predict(img_path: str, model: torch.nn.Module, device: torch.device = None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "gpu")

    img = torchvision.io.read_image(str(img_path)).type(torch.float32)
    img /= 255.

    composition = transforms.Compose([
                             transforms.Resize((64, 64))])
    
    img_transformed = composition(img)

    model.to(device)
    model.eval()
    with torch.inference_mode():

        img_pred = model(img_transformed.unsqueeze(dim=0).to(device))

        print(f"Prediction logits: {img_pred.cpu().numpy()}")

        img_pred_prob = torch.softmax(img_pred, dim=1)
        print(f"Prediction probabilities: {img_pred_prob.cpu().numpy()}")

        img_pred_label = torch.argmax(img_pred_prob, dim=1)
        print(f"Prediction label: {img_pred_label.cpu().numpy()}")

        img_pred_class = class_names[img_pred_label] # put pred label to CPU, otherwise will error
        print(f"Prediction class: {img_pred_class}")
        

def main(args):
    model = model_repo.TinyVGG( input_shape=3, hidden_units=10, output_shape=len(class_names))
    model.load_state_dict(torch.load(f=args.model_state_path))
    predict(args.img, model)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
