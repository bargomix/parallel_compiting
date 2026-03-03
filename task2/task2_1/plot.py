import re
from collections import defaultdict
import matplotlib.pyplot as plt

PATH = "results.csv"

data = defaultdict(list)   # data[N] = [(p, Tp, Sp), ...]

curN = None

with open(PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # ловим строку вида: N=20000
        m = re.match(r"^N\s*=\s*(\d+)\s*$", line)
        if m:
            curN = int(m.group(1))
            continue

        # пропускаем заголовок
        if line.lower().startswith("p,tp,sp"):
            continue

        # строки вида: 8,0.116069,3.60303
        if curN is not None:
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    p = int(parts[0].strip())
                    tp = float(parts[1].strip())
                    sp = float(parts[2].strip())
                    data[curN].append((p, tp, sp))
                except:
                    pass

if not data:
    raise SystemExit("Не нашёл данных в results.csv (ожидаю формат: N=..., потом p,Tp,Sp и строки p,Tp,Sp)")

# сортируем и фильтруем p<=40
for n in data:
    data[n] = sorted([t for t in data[n] if t[0] <= 40], key=lambda x: x[0])

# ===== Таблица =====
print("\n===== TABLE (p <= 40) =====")
for n in sorted(data.keys()):
    print(f"\nN={n}")
    print("p\tTp\t\tSp")
    print("-" * 28)
    for p, tp, sp in data[n]:
        print(f"{p}\t{tp:.6f}\t{sp:.6f}")

# ===== График =====
plt.figure()
for n in sorted(data.keys()):
    ps  = [p for p, _, _ in data[n]]
    sps = [sp for _, _, sp in data[n]]
    plt.plot(ps, sps, marker="o", label=f"N={n}")

# идеальная линия y=p (до 40)
ps_ideal = list(range(1, 41))
plt.plot(ps_ideal, ps_ideal, linestyle="--", label="Ideal")

plt.xlabel("p (threads)")
plt.ylabel("S(p)")
plt.title("Speedup up to p=40")
plt.grid(True)
plt.legend()

plt.savefig("speedup_p40.png", dpi=200, bbox_inches="tight")
plt.show()