"""Verify fresh, selected-only DeepSolo export receipts and resource evidence."""

from pathlib import Path

from scripts.icdar_budget_data import read, sha


def inspect_fresh_export(config, decision):
    spec = config.get('fresh_deepsolo')
    if spec is None:
        return dict(actual_new_deepsolo_neural_inferences=0)
    raw = Path(config['raw_test_root']); plan_path = Path(spec['plan'])
    protocol_path = Path(config['test_protocol']); prior_path = Path(spec['prior_decisions'])
    all_paths = [plan_path, protocol_path, prior_path] + [Path(p) for p in spec['receipts']]
    if not spec['receipts'] or any(config['pinned_inputs'].get(str(p)) != sha(p) for p in all_paths):
        raise ValueError('Fresh DeepSolo evidence must be explicitly pinned')
    plan = read(plan_path); prior = read(prior_path); protocol = read(protocol_path)
    if (prior['ground_truth_used'] or prior['fallback_outputs_loaded'] or
            sorted(prior['fallback_ids']) != sorted(decision['fallback_ids']) or
            sorted(prior['accepted_ids']) != sorted(decision['accepted_ids']) or
            plan['decision_sha256'] != sha(prior_path) or not plan['selected_before_inference'] or
            plan['ground_truth_paths_included'] or protocol['binding']['plan_sha256'] != sha(plan_path)):
        raise ValueError('Fresh DeepSolo must follow the identical earlier GT-free route')
    records = plan['population']['test'][spec['dataset']]
    selected = set(decision['fallback_ids'])
    if ({r['sample_id'] for r in records} != selected or len(records) != len(selected) or
            any(set(r) != {'sample_id', 'image_path', 'image_sha256'} for r in records)):
        raise ValueError('Fresh DeepSolo plan contains an unselected image or ground-truth field')
    name_to_record = {r['sample_id'].replace(':', '__') + '.json': r for r in records}
    names = set(name_to_record)
    if len(names) != len(selected):
        raise ValueError('Selected DeepSolo image filenames must be unique')
    known = {}; new_count = 0
    for receipt_path in spec['receipts']:
        receipt = read(receipt_path); files = receipt['files_sha256']
        if (receipt['status'] != 'SUCCESS' or receipt['binding'] != protocol['binding'] or
                not set(files) <= names or not set(known) <= set(files) or
                receipt['new_model_inferences'] != len(set(files) - set(known)) or
                receipt['reused'] != len(set(files) & set(known))):
            raise ValueError('Invalid selected-only DeepSolo inference receipt')
        for name, digest in files.items():
            if (name in known and known[name] != digest) or sha(raw / name) != digest:
                raise ValueError('Changed fresh DeepSolo image output')
        known.update(files); new_count += receipt['new_model_inferences']
    if set(known) != names or new_count != len(selected):
        raise ValueError('Fresh DeepSolo must infer exactly the selected images once')
    allocated = []; reserved = []; seconds = []; shapes = []
    for name in sorted(names):
        row = read(raw / name)
        contract = row['contract']; expected = name_to_record[name]
        if (row['status'] != 'SUCCESS' or row['ground_truth_used'] or
                any(contract.get(k) != v for k, v in protocol['binding'].items()) or
                contract['sample_id'] != expected['sample_id'] or contract['image_sha256'] != expected['image_sha256'] or
                row['precision'] != 'float32' or row['batch_size'] != 1 or
                row['gpu_peak_allocated_bytes'] <= 0 or
                row['gpu_peak_reserved_bytes'] < row['gpu_peak_allocated_bytes'] or row['seconds'] < 0):
            raise ValueError('Invalid fresh DeepSolo inference or GPU memory evidence')
        allocated.append(row['gpu_peak_allocated_bytes']); reserved.append(row['gpu_peak_reserved_bytes'])
        seconds.append(row['seconds']); shapes.append(row['input_chw'])
    return dict(actual_new_deepsolo_neural_inferences=new_count,
        gpu_peak_allocated_bytes=max(allocated), gpu_peak_reserved_bytes=max(reserved),
        summed_image_inference_seconds=sum(seconds), input_chw_shapes=shapes,
        precision='float32', batch_size=1, gpu=protocol['gpu'],
        scope='Selected DeepSolo subprocess image times only; no end-to-end latency claim')
