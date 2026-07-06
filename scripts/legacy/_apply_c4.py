import pathlib

p = pathlib.Path("paper/sections/conclusion.tex")
data = p.read_bytes().decode("utf-8")
crlf = "\r\n" if "\r\n" in data else "\n"
def cv(x): return x.replace("\n", crlf)

old = r"""    estimates; corrected significance tests confirm that all pairwise model differences within each sensor configuration are significant after Holm--Bonferroni correction, with LightGBM strongest throughout."""
new = r"""    estimates; corrected significance tests confirm that LightGBM significantly outperforms both linear families within each sensor configuration (Holm--Bonferroni corrected), while the two linear families are statistically indistinguishable."""

if data.count(cv(old)) != 1:
    raise SystemExit(f"matched {data.count(cv(old))} times")
data = data.replace(cv(old), cv(new))
p.write_bytes(data.encode("utf-8"))
print("OK")
