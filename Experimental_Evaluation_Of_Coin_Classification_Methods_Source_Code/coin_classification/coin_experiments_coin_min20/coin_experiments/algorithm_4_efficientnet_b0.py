"""Algorithm 4: original EfficientNet-B0 classifier, frozen or last two feature modules."""
from torch import nn
from torchvision import models
from neural_common import set_grad, train_classifier, evaluate_classifier

METHOD = 'efficientnet_b0'
CANDIDATES = ('frozen', 'partial')


def build(candidate, p, pretrained=True):
    if candidate not in CANDIDATES:
        raise ValueError(candidate)
    model = models.efficientnet_b0(weights=getattr(
        models.EfficientNet_B0_Weights,
        p['weights'][METHOD]
    ) if pretrained else None)
    set_grad(model, False)
    model.classifier = nn.Sequential(
        nn.Dropout(p['dropout']),
        nn.Linear(model.classifier[1].in_features, p['num_classes'])
    )
    unfrozen = model.features[-2:] if candidate == 'partial' else None
    if unfrozen is not None:
        set_grad(unfrozen, True)

    return model, model.features, unfrozen, model.classifier


def train(candidate, ctx, out):
    return train_classifier(build, METHOD, candidate, ctx, out)


def evaluate(ctx, candidate, out):
    return evaluate_classifier(build, ctx, candidate, out)
