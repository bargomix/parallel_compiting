import re
import matplotlib.pyplot as plt

PATH = "results.csv"

nsteps = None
T1 = None
rows = []  # (p, Tp, Sp)

with open(PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^NSTEPS=(\d+)$", line)
        if m:
            nsteps = int(m.group(1))
            continue
        m = re.match(r"^T1=([0-9.]+)$", line)
        if m:
            T1 = float(m.group(1))
            continue
        if line.lower().startswith("p,tp,sp"):
            continue
        parts = line.split(",")
        if len(parts) >= 3:
            try:
                p = int(parts[0])
                Tp = float(parts[1])
                Sp = float(parts[2])
                rows.append((p, Tp, Sp))
            except:
                pass

rows.sort(key=lambda x: x[0])

print(f"NSTEPS={nsteps} T1={T1}")
print("p\tTp\t\tSp")
print("-" * 28)
for p, Tp, Sp in rows:
    if p <= 40:
        print(f"{p}\t{Tp:.6f}\t{Sp:.6f}")

ps = [p for p, _, _ in rows if p <= 40]
sps = [Sp for p, _, Sp in rows if p <= 40]

plt.figure()
plt.plot(ps, sps, marker="o", label="Measured S(p)")
plt.plot(range(1, 41), range(1, 41), linestyle="--", label="Ideal")

plt.xlabel("p (threads)")
plt.ylabel("S(p) = T1 / Tp")
plt.title("OpenMP integrate speedup (p <= 40)")
plt.grid(True)
plt.legend()
plt.savefig("speedup_p40.png", dpi=200, bbox_inches="tight")
plt.show()