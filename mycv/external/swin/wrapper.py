import torch

from mycv.external.swin.swin_transformer import SwinTransformer


def get_swin_transformer(version: str):
    """ Get the Swin-Transformer classification model

    Args:
        version (str): one of ['tiny', 'small', 'base', 'large']

    Returns:
        torch.nn.Module: the model
    """
    assert version in ('tiny', 'small', 'base', 'large'), 'Invlaid version name'

    if version == 'tiny':
        config = {
            'DROP_PATH_RATE': 0.2,
            'EMBED_DIM': 96,
            'DEPTHS': [ 2, 2, 6, 2 ],
            'NUM_HEADS': [ 3, 6, 12, 24 ],
            'WINDOW_SIZE': 7
        }
    else:
        raise NotImplementedError()

    model = SwinTransformer(
        img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dim=config['EMBED_DIM'],
        depths=config['DEPTHS'],
        num_heads=config['NUM_HEADS'],
        window_size=config['WINDOW_SIZE'],
        mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0.0,
        drop_path_rate=config['DROP_PATH_RATE'],
        ape=False, patch_norm=True, use_checkpoint=False
    )
    return model


if __name__ == '__main__':
    model = get_swin_transformer('tiny')
    model.eval()

    # from thop import profile, clever_format
    # from fvcore.nn import flop_count
    # input = torch.randn(1, 3, 224, 224)
    # macs, params = profile(model, inputs=(input, ))
    # macs, params = clever_format([macs, params], "%.3f")
    # print(macs, params)
    # final_count, skipped_ops = flop_count(model, (input, )) 
    # print(final_count)

    from mycv.paths import MYCV_DIR
    checkpoint = torch.load(MYCV_DIR / 'weights/swin/swin_tiny_patch4_window7_224.pth')
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    from mycv.datasets.imagenet import imagenet_val
    results = imagenet_val(model, split='val',
                img_size=224, batch_size=64, workers=4, input_norm=True)
    print(results)
