
import torch



def save_model(model, PATH):
    torch.save(model,PATH)

def load_model(PATH):
    model = torch.load(PATH)
    # set to eval mode for consistency
    model.eval()
    return model
