"""Check real device and pretrained model compatibility with a disposable synthetic batch."""
import gc
import torch
from torch import nn
from config import METHODS
from common import write_json
from neural_common import training_mode, optimizer_for, seed_torch, sync
import algorithm_2_resnet50 as resnet
import algorithm_3_siamese_resnet18 as siamese
import algorithm_4_efficientnet_b0 as efficient


def device_check(out, p, num_classes, device):
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available in this PyTorch environment.')
    if device.type == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    local = dict(p, num_classes=num_classes)
    checked = []

    for module in (resnet, efficient, siamese):
        for candidate in module.CANDIDATES:
            print(
                f'Preflight: {module.METHOD}/{candidate} (ImageNet initialization + synthetic batch)',
                flush=True
            )
            seed_torch(42)
            if module is siamese:
                model, unfrozen = module.build(candidate, local, True)
                backbone = model.backbone
                head = model.embedding
            else:
                model, backbone, unfrozen, head = module.build(candidate, local, True)

            model.to(device)
            training_mode(model, backbone, unfrozen, head)
            optimizer = optimizer_for(head, unfrozen, local)
            x = torch.randn(
                local['batch_size'],
                3,
                local['image_size'],
                local['image_size'],
                device=device
            )
            optimizer.zero_grad(set_to_none=True)
            # Ensure frozen BatchNorm statistics are not updated.
            frozen_buffers = {
                n: b.detach().clone() for n, b in model.named_buffers() if 'running_' in n and not dict(model.named_modules())[n.rsplit('.', 1)[0]].training
            }
            if module is siamese:
                a, b = model(x, x.flip(-1))
                y = torch.arange(len(x), device=device) % 2
                loss = module.ContrastiveLoss(local['margin'])(a, b, y.float())
            else:
                y = torch.arange(len(x), device=device) % num_classes
                loss = nn.CrossEntropyLoss()(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite synthetic preflight loss')
            loss.backward()
            optimizer.step()
            sync(device)
            assert all(param.grad is None for param in model.parameters() if not param.requires_grad), 'Frozen gradient detected'
            for n, buf in model.named_buffers():
                if n in frozen_buffers and not torch.equal(buf, frozen_buffers[n]):
                    raise RuntimeError(f'Frozen BatchNorm updated: {n}')
            checked.append({
                'method': module.METHOD,
                'candidate': candidate,
                'total_parameters': sum(q.numel() for q in model.parameters()),
                'trainable_parameters': sum(q.numel() for q in model.parameters() if q.requires_grad)
            })
            del model, backbone, unfrozen, head, optimizer, x, y, loss, frozen_buffers
            if module is siamese:
                del a, b
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    write_json(
        out / 'device_preflight.json',
        {
            'status': 'passed',
            'device': str(device),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'batch_size': p['batch_size'],
            'image_size': p['image_size'],
            'checks': checked,
            'note': 'Disposable synthetic batch; no dataset used, no trained model retained.'
        }
    )
