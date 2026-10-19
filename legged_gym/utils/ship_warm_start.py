"""Explicit 28 -> 37 ordinary-observation migration for a nominal flat policy.

No optimizer state is reused. The 9 new deck-IMU input columns start at zero,
preserving the existing controller and GRU response until fine-tuning.
"""
import torch


def _copy_with_input_map(old, template, mapping):
    if old.shape[0] != template.shape[0] or old.shape[1] != len(mapping):
        raise ValueError("incompatible layer width for ship input migration")
    value = torch.zeros_like(template)
    value[:, mapping] = old.to(device=value.device, dtype=value.dtype)
    return value


def migrate_flat_checkpoint(checkpoint, current_model, current_encoder):
    old_model = checkpoint["model_state_dict"]
    old_encoder = checkpoint["encoder_state_dict"]
    if old_model["actor.0.weight"].shape[1] != 38:
        raise ValueError("warm start requires 28 ordinary obs + 7 latent + 3 commands")
    if old_model["critic.0.weight"].shape[1] != 45:
        raise ValueError("warm start requires the nominal 7-dim critic prefix")
    if set(old_model) != set(current_model) or set(old_encoder) != set(current_encoder):
        raise ValueError("source and target network structures differ")
    model, encoder = {}, {}
    for key, template in current_model.items():
        source = old_model[key]
        if key in ("actor.0.weight", "critic.0.weight"):
            if template.shape[1] != source.shape[1]+9:
                raise ValueError("target policy must have exactly 9 added deck inputs")
            mapping = list(range(27)) + list(range(36,template.shape[1]))
            model[key] = _copy_with_input_map(source,template,mapping)
        else:
            if source.shape != template.shape:
                raise ValueError(f"incompatible source model parameter: {key}")
            model[key] = source.to(device=template.device,dtype=template.dtype)
    for key, template in current_encoder.items():
        source = old_encoder[key]
        if key.endswith("gru.weight_ih_l0"):
            if source.shape[1] != 48 or template.shape[1] != 66:
                raise ValueError("GRU migration requires raw+filtered frame widths 48 -> 66")
            # raw: 20 state + 8 actions; filtered: 20 state. Target inserts
            # IMU before raw actions and appends IMU to the filtered state.
            mapping = list(range(20)) + list(range(29,57))
            encoder[key] = _copy_with_input_map(source,template,mapping)
        else:
            if source.shape != template.shape:
                raise ValueError(f"incompatible source encoder parameter: {key}")
            encoder[key] = source.to(device=template.device,dtype=template.dtype)
    return model, encoder


def migrate_ship_mobility_checkpoint(checkpoint, current_model, current_encoder):
    """37 -> 40 obs, preserving a trained ship policy at initialization.

    Insert requested gait mode/sin/cos before the previous eight actions.
    Both raw and filtered GRU branches gain three zero-weight columns. Never
    reuse optimizer state or accidentally treat a flat/gait policy as a ship one.
    """
    old_model, old_encoder = checkpoint['model_state_dict'], checkpoint['encoder_state_dict']
    if old_model['actor.0.weight'].shape[1] != 47 or old_model['critic.0.weight'].shape[1] != 54:
        raise ValueError('mobility warm start requires a 37-observation ship policy')
    if set(old_model) != set(current_model) or set(old_encoder) != set(current_encoder):
        raise ValueError('source and target network structures differ')
    model, encoder = {}, {}
    for key, template in current_model.items():
        source = old_model[key]
        if key in ('actor.0.weight', 'critic.0.weight'):
            if template.shape[1] != source.shape[1]+3:
                raise ValueError('mobility policy must add exactly three gait columns')
            model[key] = _copy_with_input_map(source, template,
                list(range(36))+list(range(39, template.shape[1])))
        else:
            if source.shape != template.shape:
                raise ValueError('incompatible source model parameter: '+key)
            model[key] = source.to(device=template.device, dtype=template.dtype)
    for key, template in current_encoder.items():
        source = old_encoder[key]
        if key.endswith('gru.weight_ih_l0'):
            if source.shape[1] != 66 or template.shape[1] != 72:
                raise ValueError('mobility GRU frame must expand from 66 to 72')
            encoder[key] = _copy_with_input_map(source, template,
                list(range(29))+list(range(32, 69)))
        else:
            if source.shape != template.shape:
                raise ValueError('incompatible source encoder parameter: '+key)
            encoder[key] = source.to(device=template.device, dtype=template.dtype)
    return model, encoder


def ship_finetune_state_dicts(checkpoint, current_model, current_encoder):
    """Strict same-layout ship continuation; weights only, not optimizer/iteration."""
    old_model, old_encoder = checkpoint['model_state_dict'], checkpoint['encoder_state_dict']
    if old_model['actor.0.weight'].shape[1] != 47 or old_model['critic.0.weight'].shape[1] != 54:
        raise ValueError('ship finetune requires actor47/critic54, not flat or mobility policy')
    if not any(k.endswith('gru.weight_ih_l0') and v.shape[1] == 66 for k, v in old_encoder.items()):
        raise ValueError('ship finetune requires the 66-feature ship GRU frame')
    result = []
    for source, target in [(old_model, current_model), (old_encoder, current_encoder)]:
        if set(source) != set(target):
            raise ValueError('ship finetune network keys differ')
        values = {}
        for key, template in target.items():
            value = source[key]
            if value.shape != template.shape:
                raise ValueError('ship finetune parameter shape differs: '+key)
            if not torch.isfinite(value).all():
                raise ValueError('nonfinite ship finetune parameter: '+key)
            values[key] = value.to(device=template.device, dtype=template.dtype).clone()
        result.append(values)
    return tuple(result)


def ship_prior_state_dicts(checkpoint, current_model, current_encoder):
    """Standing ship37 -> prior61, or strict prior61 continuation, weights only.

    New actor/critic AND raw/filtered GRU input columns are zero initialized.
    Preserve old forward behavior for corresponding inputs, not optimizer or
    rollout RNG state. Never silently migrate a flat or mobility checkpoint.
    """
    old_model, old_encoder = checkpoint['model_state_dict'], checkpoint['encoder_state_dict']
    width = old_model['actor.0.weight'].shape[1]
    if width not in (47, 71) or old_model['critic.0.weight'].shape[1] != width+7:
        raise ValueError('prior warm start requires standing ship actor47/critic54 or prior actor71/critic78')
    if set(old_model) != set(current_model) or set(old_encoder) != set(current_encoder):
        raise ValueError('prior warm start network keys differ')
    expand = width == 47
    result = []
    for kind, source, target in [('model', old_model, current_model), ('encoder', old_encoder, current_encoder)]:
        values = {}
        for key, template in target.items():
            value = source[key]
            if not torch.isfinite(value).all():
                raise ValueError('nonfinite prior warm-start parameter: '+key)
            if expand and kind == 'model' and key in ('actor.0.weight', 'critic.0.weight'):
                if template.shape[1] != value.shape[1]+24:
                    raise ValueError('prior migration adds exactly 24 ordinary features')
                values[key] = _copy_with_input_map(value, template,
                    list(range(36))+list(range(60, template.shape[1])))
            elif expand and kind == 'encoder' and key.endswith('gru.weight_ih_l0'):
                if value.shape[1] != 66 or template.shape[1] != 114:
                    raise ValueError('prior migration requires GRU frame66 -> frame114')
                values[key] = _copy_with_input_map(value, template,
                    list(range(29))+list(range(53, 90)))
            else:
                if value.shape != template.shape:
                    raise ValueError('prior parameter shape differs: '+key)
                values[key] = value.to(device=template.device, dtype=template.dtype).clone()
        result.append(values)
    if (current_model['actor.0.weight'].shape[1] != 71
        or current_model['critic.0.weight'].shape[1] != 78
        or not any(k.endswith('gru.weight_ih_l0') and v.shape[1] == 114 for k, v in current_encoder.items())):
        raise ValueError('target must use prior actor71/critic78/GRU114')
    return tuple(result)
