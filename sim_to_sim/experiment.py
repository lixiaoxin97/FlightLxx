#!/usr/bin/env python3
"""固定 PID 的初态对照实验：生成、运行、查看、分析和数据核验。"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import random
import stat as stat_types
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"runtime"))
from result_io import digest, save

A = HERE/"logs/continued_pid_20261029_v1/rounds/r005/formal_continue_736ee51a5cf0/trials_batch"
B = HERE/"logs/full_random_pid_20261030_v1/evaluation"
DATASETS = {"curriculum": A, "random": B}
PROTECTED = (A.parent, B, B.parent/"pilot")
MODELS = ("CTBR", "CTBR+DR", "CTBR+ID")


def bank_path(value):
    if value in DATASETS:
        return HERE/"states"/(value+"_1000.json")
    return Path(value).expanduser().resolve()


def writable_output(path):
    path = Path(path).expanduser().resolve()
    if any(path == p or path.is_relative_to(p) or p.is_relative_to(path) for p in PROTECTED):
        raise ValueError("保留的两轮原始数据为只读，请使用新的输出目录")
    return path


def status(folder=None):
    from analysis import collect
    targets = {str(folder): Path(folder)} if folder else DATASETS
    for label, root in targets.items():
        manifest, rows = collect(root)
        print(f"{label}: {len(rows)}/{len(manifest['schedule'])} completed")
        for model in MODELS:
            group = [r for r in rows if r["model"] == model]
            counts = Counter(r.get("failure_category") or "success" for r in group)
            print(f"  {model}: {counts['success']}/{len(group)}; 初始化失败 {counts['initialization_failure']}, "
                  f"通信/模式中断 {counts['communication_mode_interruption']}, 策略失败 {counts['policy_recovery_failure']}")


def verify(raw_files=False):
    from analysis import collect
    from fixed_pid import PROFILES, PROFILE
    audit = json.loads((HERE/"reports/integrity/completed_sha256.json").read_text())
    audit.update(json.loads((HERE/"reports/integrity/retained_sha256.json").read_text()))
    for name, sha in audit.items():
        path = Path(name) if Path(name).is_absolute() else HERE/name
        if not path.is_file() or digest(path) != sha:
            raise RuntimeError("数据哈希不一致: " + str(path))
    pid = {k:float(v) for k,v in PROFILES[PROFILE].items()}
    for label, expected in (("curriculum", (374,480,599)), ("random", (297,419,540))):
        manifest, rows = collect(DATASETS[label])
        states = {s["id"]: s["state"] for s in json.loads(bank_path(label).read_text())["states"]}
        if len(rows) != 3000 or len(manifest["schedule"]) != 3000:
            raise RuntimeError("原始实验数量不完整")
        counts = []
        for model in MODELS:
            group = [r for r in rows if r["model"] == model]
            if len(group) != 1000 or {r["state_id"] for r in group} != set(states):
                raise RuntimeError("网络初态列表不一致")
            for row in group:
                if row["spec"]["state"] != states[row["state_id"]]:
                    raise RuntimeError("请求初态被改变")
                if row["valid_policy_trial"]:
                    if not row["effective_pid_verified"] or any(abs(row["effective_pid"][k]-v)>2e-6 for k,v in pid.items()):
                        raise RuntimeError("有效飞行的实际 PID 不一致")
                    if any(not 0<=v<.06 for key in ("initialization_errors", "action_input_errors") for v in row[key].values()):
                        raise RuntimeError("初始化容差不一致")
            counts.append(sum(r["success"] for r in group))
        if tuple(counts) != expected:
            raise RuntimeError("原始成功次数不一致")
        print(label, dict(zip(MODELS, counts)), "verified")
    if raw_files:
        with gzip.open(HERE/"reports/integrity/raw_inventory.json.gz", "rt") as stream:
            inventory = json.load(stream)
        for name, previous in inventory.items():
            path = HERE/name
            stat = path.lstat()
            kind = "symlink" if stat_types.S_ISLNK(stat.st_mode) else "file" if stat_types.S_ISREG(stat.st_mode) else "other"
            current = dict(type=kind, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if kind == "other":
                current["mode"] = stat.st_mode
            if path.is_symlink():
                current["target"] = os.readlink(path)
            if current != previous:
                raise RuntimeError("原始文件属性改变: " + name)
        print(f"{len(inventory)} raw files/symlinks unchanged (size, mtime, link targets)")
    print("6000份结果哈希、成对初态、实际PID与最终报告核验通过。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    info = commands.add_parser("status", help="读取两轮保留结果或指定的新实验")
    info.add_argument("--output", type=Path)
    check = commands.add_parser("verify", help="只读核验两轮原始结果")
    check.add_argument("--raw-files", action="store_true", help="另核对所有原始文件大小、mtime和链接目标")
    generate = commands.add_parser("generate", help="按既定范围全随机生成初态")
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--count", type=int, default=1000)
    generate.add_argument("--output", type=Path, required=True)
    for name in ("run", "view"):
        cmd = commands.add_parser(name, help="三网络对照实验" if name=="run" else "逐场启动Gazebo，观察相同初态")
        cmd.add_argument("--bank", default="random", help="random、curriculum或自定义初态JSON路径")
        cmd.add_argument("--output", type=Path, required=name=="run")
        cmd.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
        cmd.add_argument("--max-trials", type=int)
        cmd.add_argument("--plan-only", action="store_true")
        if name == "run":
            cmd.add_argument("--workers", type=int, choices=range(1,13), default=12)
            cmd.add_argument("--resume", action="store_true")
        else:
            cmd.add_argument("--count", type=int, default=10, help="抽取初态数，每网络使用同一列表")
            cmd.add_argument("--seed", type=int, default=2026103001)
            cmd.add_argument("--indices", nargs="+", type=int, help="指定初态序号，从1开始，替代随机抽取")
            cmd.add_argument("--pause", type=float, default=3., help="场间等待秒数")
    analysis = commands.add_parser("analyze", help="为新实验生成CSV、统计图、Markdown和PDF")
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    if args.action == "status":
        status(args.output)
    elif args.action == "verify":
        verify(args.raw_files)
    elif args.action == "generate":
        from state_bank import generate
        dest = writable_output(args.output)
        if dest.exists():
            parser.error("初态文件已存在，请选择新路径")
        save(dest, generate(args.seed, args.count, "random_"+str(args.seed)))
        print(dest)
    elif args.action == "analyze":
        from analysis import build
        result = build(writable_output(args.output), not args.no_pdf)
        print(f"分析完成: {result['completed']}/{result['planned']}")
    else:
        from runner import run_batch, schedule
        source = bank_path(args.bank)
        bank = json.loads(source.read_text())
        if args.max_trials is not None and args.max_trials < 1:
            parser.error("max-trials必须为正整数")
        if args.action == "view":
            if args.pause < 0:
                parser.error("pause不能为负数")
            if args.indices:
                if len(set(args.indices)) != len(args.indices) or any(i<1 or i>len(bank["states"]) for i in args.indices):
                    parser.error("indices必须为不重复的有效初态序号")
                states = [bank["states"][i-1] for i in args.indices]
            else:
                if not 1<=args.count<=len(bank["states"]):
                    parser.error("count超出初态数量")
                states = random.Random(args.seed).sample(bank["states"], args.count)
            bank = dict(bank, states=states, count=len(states), view_selection_seed=args.seed,
                        purpose="New live demonstration; excluded from preserved formal results")
        plan = schedule(bank, args.models)
        if args.plan_only:
            print(json.dumps(plan[:args.max_trials] if args.max_trials else plan, ensure_ascii=False, indent=2))
            return
        dest = writable_output(args.output or HERE/"logs"/("view_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
        if args.action == "view":
            if not os.environ.get("DISPLAY") or not Path("/tmp/.X11-unix").is_dir():
                parser.error("Gazebo可视化需要当前桌面的X11会话")
            if dest.exists():
                parser.error("请选择新的可视化输出目录")
            dest.mkdir(parents=True)
            source = dest/"view_states.json"
            save(source, bank)
        seed = B/"parameter_seed" if args.bank != "curriculum" else A/"parameter_seed"
        run_batch(source, dest, workers=1 if args.action=="view" else args.workers,
                  models=args.models, max_trials=args.max_trials, parameter_seed=seed,
                  resume=args.resume if args.action=="run" else False, gui=args.action=="view",
                  pause=args.pause if args.action=="view" else 0.)


if __name__ == "__main__":
    main()
