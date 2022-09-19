allModels = ["unet", "ib_unet_k3", "ib_unet_k5", 
            "attention_unet", "ib_att_unet_k3", "ib_att_unet_k5", 
            "segresnet", "ib_segresnet_k3", "ib_segresnet_k5"]

def get_model(opt=None):
    """
    Retrieves the model architecture based on the requirement.
    Args: model_name: Must be a string from choice list.
          opt: Command line argument.
    """
    # Original UNet
    if opt.model_name == "unet":
        from networks.unet import UNet as model_class

    # IB UNet k=3
    elif opt.model_name == "ib_unet_k3":
        from networks.ib_unet import IB_UNet_k3 as model_class

    # IB UNet k=5
    elif opt.model_name == "ib_unet_k5":
        from networks.ib_unet import IB_UNet_k5 as model_class

    # Attention-UNet
    elif opt.model_name == "attention_unet":
        from networks.attention_unet import Attention_UNet as model_class

    # IB Attention U-Net K=3
    elif opt.model_name == "ib_att_unet_k3":
        from networks.ib_att_unet import IB_Attention_UNet_k3 as model_class

    # IB Attention U-Net K=5
    elif opt.model_name == "ib_att_unet_k5":
        from networks.ib_att_unet import IB_Attention_UNet_k5 as model_class

    # Seg ResNet
    elif opt.model_name == "segresnet":
        from monai.networks.nets import SegResNet 
        model = SegResNet(out_channels=opt.output_channels, init_filters=opt.n_kernels, dropout_prob=opt.dropout_rate)
        return model

    # IB Seg ResNet K=3
    elif opt.model_name == "ib_segresnet_k3":
        from networks.ib_segresnet import IB_SegResNet_k3 
        model = IB_SegResNet_k3(out_channels=opt.output_channels, init_filters=opt.n_kernels, dropout_prob=opt.dropout_rate, device=opt.device_id)
        return model

    # IB Seg ResNet K=5
    elif opt.model_name == "ib_segresnet_k5":
        from networks.ib_segresnet import IB_SegResNet_k5 
        model = IB_SegResNet_k5(out_channels=opt.output_channels, init_filters=opt.n_kernels, dropout_prob=opt.dropout_rate, device=opt.device_id)
        return model

    else:
        raise Exception("Re-check the model name, otherwise the model isn't available.")
    
    model = model_class(opt)
    return model

