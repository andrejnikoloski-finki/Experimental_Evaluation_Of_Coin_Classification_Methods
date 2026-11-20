"""Algorithm 2: original ResNet50 classifier, frozen or layer4 fine-tuning."""
from torch import nn
from torchvision import models
from neural_common import set_grad, train_classifier, evaluate_classifier

METHOD = 'resnet50'
CANDIDATES = ('frozen', 'partial')


def build(candidate, p, pretrained=True):
    if candidate not in CANDIDATES:
        raise ValueError(candidate)
    model = models.resnet50(weights=getattr(
        models.ResNet50_Weights,
        p['weights'][METHOD]
    ) if pretrained else None)
    set_grad(model, False)
    model.fc = nn.Sequential(
        nn.Dropout(p['dropout']),
        nn.Linear(model.fc.in_features, p['num_classes'])
    )
    unfrozen = model.layer4 if candidate == 'partial' else None
    if unfrozen is not None:
        set_grad(unfrozen, True)

    return model, model, unfrozen, model.fc


def train(candidate, ctx, out):
    return train_classifier(build, METHOD, candidate, ctx, out)


def evaluate(ctx, candidate, out):
    return evaluate_classifier(build, ctx, candidate, out)
