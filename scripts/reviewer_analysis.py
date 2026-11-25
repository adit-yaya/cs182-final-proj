"""Aggregate dense snapshots and draw the width-256 figures."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from plot_empirical_transfer import ideal_response, plt
from run_w256 import NS_PE, TIKH, ROOT

METHODS = list(NS_PE) + list(TIKH)
OPS = ["Attention QKV", "Attention output", "MLP expansion", "MLP contraction"]
LAYERS = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
STEPS = [1, 10, 50, 100, 250, 300, 350, 400, 450, 500]
CUT = dict(ns5=.002480860233540957, pe3=.008193073987001599, pe5=.001317823050864259)
BINS = np.geomspace(1e-6, 1.6, 76)
COLORS = ["black", "#888888", "#75418a", "steelblue", "C0", "C1", "C2", "C3"]
LABELS = ["NS5", "PE3", "PE5", "Exact polar (λ = 0)", "λ = 1e-8", "λ = 1e-6", "λ = 1e-4", "λ = 1e-2"]
REFERENCE = ROOT / "results/w256_bpe_dense"


def collect(folder):
    shape = (len(METHODS), 3, 4, 6, len(STEPS))
    metrics = np.full((*shape, 6), np.nan)
    hist = np.zeros((*shape, 2, 75), dtype=np.uint16)
    train = np.empty((8, 3, 500))
    val = np.empty((8, 3, 5))
    for m, method in enumerate(METHODS):
        for seed in range(3):
            path = folder / f"{method}_seed{24+seed}"
            record = json.loads((path / "results.json").read_text())
            train[m, seed] = record["losses"]
            val[m, seed] = [row["loss"] for row in record["validation_losses"]]
            for snap in torch.load(path / "momenta.pt", weights_only=True, map_location="cpu", mmap=True):
                _, _, block, group, layer, _ = snap["layer"].split(".")
                key = (m, seed, LAYERS.index(group+"."+layer), int(block), STEPS.index(snap["step"]))
                x, y = (snap[k].numpy() for k in ("normalized_sv", "output_sv"))
                metrics[key][:4] = [100*np.mean(y<.8), np.median(y), 100*np.mean(y<.2), 100*np.mean(y>1.2)]
                if method in CUT:
                    low = abs(snap["projection"].numpy()) < .8
                    before = x < CUT[method]
                    metrics[key][4:] = [100*np.mean(low & before), 100*np.mean(low & ~before)]
                for i, values in enumerate((x, y)):
                    hist[key][i] = np.histogram(np.maximum(values, BINS[0]), bins=BINS)[0]
    assert np.isfinite(metrics[..., :4]).all()
    return dict(metrics=metrics, hist=hist, train=train, val=val)


def table(out, name, rows):
    with (REFERENCE / name).open() as source:
        header = next(csv.reader(source))
    with (out / name).open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        writer.writerows(rows)


def save(fig, out, name):
    fig.savefig(out / (name+".png"), dpi=180)
    plt.close(fig)


def analyze(out, use_runs=False):
    out.mkdir(parents=True, exist_ok=True)
    data = collect(out) if use_runs else np.load(ROOT / "data/dense_summary.npz")
    metrics, hist, train, val = (data[k] for k in ("metrics", "hist", "train", "val"))
    rows, summary = [], []
    for m, method in enumerate(METHODS):
        values = metrics[m].mean(axis=0)
        for op, block, t in np.ndindex(values.shape[:3]):
            row = values[op, block, t]
            rows.append([method, OPS[op], block+1, STEPS[t], *[x if np.isfinite(x) else "" for x in row]])
        avg = [values[..., column].mean() for column in range(values.shape[-1])]
        delta = val[m,:,-1] - val[0,:,-1]
        summary.append([method, avg[1], avg[0], avg[5] if method in CUT else "", val[m,:,-1].mean(),
                        delta.mean(), delta.std(ddof=1)])
    table(out, "orthogonalization_metrics.csv", rows)
    table(out, "orthogonalizer_comparison.csv", summary)
    loss_plots(out, train, val)
    for m in [0, 1, 2, 4, 6]:
        for facet in ("depth", "step"):
            fig, ax = plt.subplots(figsize=(8,4.6), layout="constrained")
            depth = facet == "depth"
            xs = range(1,7) if depth else STEPS
            for op, label in enumerate(OPS):
                y = metrics[m,:,op,:,4,0] if depth else metrics[m,:,op,2,:,0]
                mean, sd = y.mean(0), y.std(0,ddof=1)
                line, = ax.plot(xs, mean, "o-" if op<2 else "o--", label=label, ms=4)
                ax.fill_between(xs, mean-sd, mean+sd, color=line.get_color(), alpha=.14)
            ax.set(title=LABELS[m]+" at "+("step 250" if depth else "block 3"),
                   xlabel="Block" if depth else "Training step", ylabel="Output singular values below 0.8 (%)",
                   ylim=(0,100), xticks=xs if depth else range(0,501,50))
            ax.legend(ncol=2, fontsize=9)
            ax.grid(alpha=.2)
            save(fig,out,f"{METHODS[m]}_below_0_8_by_{facet}")
    for m in (0,2):
        for facet in ("depth","step"):
            fig, axes = plt.subplots(4,3,figsize=(9,7),sharex=True,sharey=True,layout="constrained")
            pairs = [(0,4),(2,4),(5,4)] if facet=="depth" else [(2,1),(2,4),(2,9)]
            for op, axes_row in enumerate(axes):
                for ax,(b,t) in zip(axes_row,pairs):
                    pre, post = hist[m,:,op,b,t].mean(0)*100/256
                    ax.stairs(post,BINS,color=COLORS[m],fill=True,alpha=.28,label="After")
                    ax.stairs(post,BINS,color=COLORS[m])
                    ax.stairs(pre,BINS,color="#777777",label="Before")
                    ax.axvline(1,color=".6",ls=":",lw=.8)
                    ax.set(xscale="log",xlim=(BINS[0],BINS[-1]),ylim=(0,64))
                    if op==0: ax.set_title(f"Block {b+1}" if facet=="depth" else f"Step {STEPS[t]}")
                axes_row[0].set_ylabel(OPS[op])
            axes[0,0].legend(fontsize=8)
            fig.suptitle(LABELS[m]+" at "+("step 250" if facet=="depth" else "block 3"))
            fig.supxlabel("Singular value")
            fig.supylabel("Singular values per bin (%)")
            save(fig,out,f"{METHODS[m]}_spectra_by_{facet}")
    fig,ax=plt.subplots(figsize=(9,6),layout="constrained")
    x=np.geomspace(1e-5,1,5000)
    for i,m in enumerate(METHODS):
        if i<3: y=ideal_response(x,m[:2],int(m[2]))
        else: y=x/np.sqrt(x*x+float(m.split("_")[1]))
        ax.plot(x,y,color=COLORS[i],ls="-" if i<4 else "--",label=LABELS[i])
    ax.set(xscale="log",xlabel="Frobenius-normalized input singular value",ylabel="Output singular value")
    ax.legend()
    ax.grid(which="both",alpha=.22)
    save(fig,out,"transfer_curves")


def loss_plots(out, train, val):
    variants = [("training_loss_comparison",25,"none"),("training_loss_comparison_std",25,"bars"),
                ("training_loss_comparison_std_bands",25,"bands"),("training_loss_comparison_raw_std_bands",1,"bands"),
                ("validation_loss_comparison_std_bands",1,"bands")]
    for name,window,style in variants:
        validation=name.startswith("validation")
        fig,ax=plt.subplots(figsize=(9,5),layout="constrained")
        for m,losses in enumerate(val if validation else train):
            smooth=np.array([np.convolve(v,np.ones(window)/window,"valid") for v in losses])
            mean,sd=smooth.mean(0),smooth.std(0,ddof=1)
            x=np.arange(100,501,100) if validation else np.arange(window,501)
            ax.plot(x,mean,color=COLORS[m],ls="-" if m<4 else "--",label=LABELS[m],marker="o" if validation else None,ms=3)
            if style=="bands": ax.fill_between(x,mean-sd,mean+sd,color=COLORS[m],alpha=.12,lw=0)
            if style=="bars": ax.errorbar(x,mean,yerr=sd,fmt="none",errorevery=(3*m,25),ecolor=COLORS[m],capsize=2)
        ax.set(xlabel="Training step",ylabel="Validation loss" if validation else "Training loss")
        ax.legend(ncol=2,fontsize=9)
        ax.grid(alpha=.2)
        save(fig,out,name)
