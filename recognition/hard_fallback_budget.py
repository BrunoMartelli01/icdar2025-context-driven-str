"""Explicit image-budget constraints, separate from scientific STRICT mode."""


def validate_policy(contracts, dataset, policy_quota):
    if dataset not in contracts:
        raise ValueError('Missing explicit hard fallback budget for '+dataset)
    contract = contracts[dataset]
    if set(contract) != {'reference_images', 'maximum_fallback_images', 'batch_rounding'}:
        raise ValueError('Explicit reference population, maximum and rounding required')
    n, maximum = contract['reference_images'], contract['maximum_fallback_images']
    if (type(n) is not int or type(maximum) is not int or n < 1 or
            not 0 <= maximum <= n or contract['batch_rounding'] != 'floor'):
        raise ValueError('Valid hard fallback budget with floor rounding required')
    if type(policy_quota) is not int or not 0 <= policy_quota <= maximum:
        raise ValueError('Requested policy exceeds the hard fallback budget')
    return dict(contract)


def batch_quota(contract, policy_quota, reference_images, images):
    # Revalidate standalone callers; booleans must not act as integer budgets.
    validate_policy({'batch': contract}, 'batch', policy_quota)
    if type(reference_images) is not int or reference_images != contract['reference_images']:
        raise ValueError('Hard budget reference differs from the frozen policy population')
    if type(images) is not int or images < 1:
        raise ValueError('Nonempty integer batch population required')
    return policy_quota*images//reference_images


def verify_route(decision, sample_ids, contract, policy_quota):
    ids = set(sample_ids)
    accepted, fallback = decision['accepted_ids'], decision['fallback_ids']
    expected = batch_quota(contract, policy_quota, contract['reference_images'], len(sample_ids))
    if (len(ids) != len(sample_ids) or len(accepted) != len(set(accepted)) or
            len(fallback) != len(set(fallback)) or set(accepted) & set(fallback) or
            set(accepted) | set(fallback) != ids or len(fallback) != expected or
            decision['deepsolo_images'] != expected or decision['total_images'] != len(ids)):
        raise ValueError('Routing decision violates the hard fallback budget or population')
    if abs(decision['deepsolo_percentage']-100*expected/len(ids)) > 1e-10:
        raise ValueError('Hard fallback routing percentage differs from its image counts')
