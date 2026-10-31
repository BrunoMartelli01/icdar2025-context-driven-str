"""Sequential validation/selection/TEST after the matched training job exits."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from segmentation.data import file_sha256
from segmentation.icdar2015_improvement_runtime import load_config
from scripts.train_icdar2015_totaltext_transfer import write_json


def process_identity(pid):
    try:return Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[19]
    except FileNotFoundError:return None


def run(args):
    config=load_config(args.config);root=args.suite_root.resolve()
    logfile=root/"evaluation-suite.log"
    def command(module,flags):
        cmd=[sys.executable,"-u","-m",module,*[str(x) for x in flags]]
        with logfile.open("a") as stream:
            stream.write(json.dumps({"command":cmd,"utc":datetime.now(timezone.utc).isoformat()})+"\n");stream.flush()
            process=subprocess.Popen(cmd,cwd=args.repo_root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:
                print(line,end="",flush=True);stream.write(line);stream.flush()
            code=process.wait()
            if code:raise RuntimeError(f"{module} exited with status {code}; see {logfile}")
    if args.training_pid is not None:
        identity=process_identity(args.training_pid)
        if identity is None:raise ValueError("Specified training supervisor is no longer live")
        write_json(root/"evaluation_training_dependency.json",{"pid":args.training_pid,"start_time_ticks":identity,
            "created_utc":datetime.now(timezone.utc).isoformat(),"source_sha256":file_sha256(Path(__file__))})
        print(f"[evaluation suite] waiting for verified training supervisor {args.training_pid}",flush=True)
        while process_identity(args.training_pid)==identity:time.sleep(10)
    for name in config["branches"]:
        report=json.loads((root/name/"run_manifest.json").read_text())
        if report["status"]!="SUCCESS" or not report["training_complete"] or report["completed_epochs"]!=config["training"]["epochs"]:
            raise ValueError("All four complete matched training runs are required")
    if not args.device.startswith("cuda"):
        raise ValueError("Final matched validation/TEST uses an explicit CUDA device")
    hardware=subprocess.run(["nvidia-smi","--query-gpu=name,uuid,driver_version,memory.total,memory.used",
        "--format=csv"],check=True,capture_output=True,text=True)
    write_json(root/"evaluation_hardware.json",{"utc":datetime.now(timezone.utc).isoformat(),"nvidia_smi":hardware.stdout})
    validation=root/"validation-gpu"
    common=["--config",args.config,"--data-root",args.data_root,"--parseq-source",args.parseq_source,
        "--parseq-source-archive",args.parseq_source_archive,"--parseq-checkpoint",args.parseq_checkpoint,"--device",args.device]
    names=["source_square512","source_square768",*config["branches"]]
    for name in names:
        origin=(["--screening-dir",root/"resolution-screen","--screen-resolution",name.removeprefix("source_")]
            if name.startswith("source_") else ["--training-run",root/name])
        command("scripts.evaluate_icdar2015_improvement_validation",[*common,*origin,"--output-dir",validation/name])
        command("scripts.audit_icdar2015_improvement_word_f1",["--results",validation/name/"results.json",
            "--data-root",args.data_root,"--output",validation/name/"independent_audit.json"])
    differences={}
    for name in names:
        cpu=root/"qualification-validation-ocr-cpu-v2"/name/"results.json"
        if not cpu.exists():continue
        a=json.loads(cpu.read_text());b=json.loads((validation/name/"results.json").read_text())
        old={p["crop_rgb_sha256"]:p["text"] for row in a["per_image"] for p in row["predictions"]}
        new={p["crop_rgb_sha256"]:p["text"] for row in b["per_image"] for p in row["predictions"]}
        if set(old)!=set(new):raise ValueError("CPU/GPU OCR qualification crop populations differ")
        differences[name]={"crops":len(old),"different_text_outputs":sum(old[k]!=new[k] for k in old),
            "aggregate_identical":a["aggregate"]==b["aggregate"],"cpu_result_sha256":file_sha256(cpu),
            "gpu_result_sha256":file_sha256(validation/name/"results.json")}
    write_json(root/"cpu_gpu_ocr_comparison.json",{"status":"COMPARED","comparisons":differences,
        "policy":"GPU validation selects the model regardless of CPU qualification differences."})
    selection=root/"selection.json"
    command("scripts.select_icdar2015_improvement",["--config",args.config,"--suite-root",root,
        "--validation-root",validation,"--output",selection])
    selected=json.loads(selection.read_text())["selected_branch"];test=root/"selected-test500"
    flags=["--config",args.test_config,"--training-config",args.config,"--training-run",root/selected,
        "--selection-report",selection,"--test-manifest",args.test_manifest,"--data-root",args.test_data_root,
        "--parseq-source",args.parseq_source,"--parseq-source-archive",args.parseq_source_archive,
        "--parseq-checkpoint",args.parseq_checkpoint,"--output-dir",test,"--device",args.device]
    command("scripts.evaluate_icdar2015_improvement_test",[*flags,"--preflight-only"])
    command("scripts.evaluate_icdar2015_improvement_test",flags)
    command("scripts.audit_icdar2015_improvement_word_f1",["--results",test/"results.json",
        "--data-root",args.test_data_root,"--output",test/"independent_audit.json"])
    # Completion reopening must preserve all cached scientific outputs.
    command("scripts.evaluate_icdar2015_improvement_test",flags)
    write_json(root/"evaluation_suite.json",{"status":"SUCCESS","reproduction_mode":"approximate",
        "claim_label":"approximate reproduction","selected_branch":selected,
        "selection_sha256":file_sha256(selection),"test_result_sha256":file_sha256(test/"results.json"),
        "independent_audit_sha256":file_sha256(test/"independent_audit.json"),
        "source_sha256":file_sha256(Path(__file__)),"completed_utc":datetime.now(timezone.utc).isoformat()})
    print(f"[evaluation suite] complete; selected {selected}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","test-config","repo-root","suite-root","data-root","test-data-root","test-manifest",
                 "parseq-source","parseq-source-archive","parseq-checkpoint"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device",required=True);parser.add_argument("--training-pid",type=int)
    args=parser.parse_args()
    try:run(args)
    except Exception as error:
        write_json(args.suite_root/"evaluation_suite_failure.json",{"status":"FAILED","error_type":type(error).__name__,"error":str(error)})
        raise


if __name__=="__main__":main()
