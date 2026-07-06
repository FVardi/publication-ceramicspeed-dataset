import pathlib

p = pathlib.Path("paper/sections/conclusion.tex")
data = p.read_bytes().decode("utf-8")
crlf = "\r\n" if "\r\n" in data else "\n"
def cv(x): return x.replace("\n", crlf)

# Finding 2: fusion gain is no longer "small" -- match the reframed body.
old2 = r"""  \item \textbf{Sensor fusion yields a small but consistently significant gain.}
    The LightGBM combined model achieves $R^2 = \resHOrsqLgbComb$ ($\Delta R^2 = \resDrsqLgbCombAe$ over AE alone); within-CV corrected $t$-test $p = \resPcvAeComb$; hold-out tests: Wilcoxon $p = \resPwxAeComb$, Diebold--Mariano $p = \resPdmAeComb$, bootstrap 95\%~CI $[\resDrmseLoAeComb,\, \resDrmseHiAeComb]$ excluding zero. The heterodyned US channel thus supplies a modest amount of non-redundant information, though its \resNretUs{} retained features are individually weaker than the AE features."""
new2 = r"""  \item \textbf{Sensor fusion yields a substantial, consistently significant gain.}
    The LightGBM combined model achieves $R^2 = \resHOrsqLgbComb$ ($\Delta R^2 = \resDrsqLgbCombAe$ over AE alone---a relative RMSE reduction of about \resRelRmseRedCombAe\%); within-CV corrected $t$-test $p = \resPcvAeComb$; hold-out tests: Wilcoxon $p = \resPwxAeComb$, Diebold--Mariano $p = \resPdmAeComb$, bootstrap 95\%~CI $[\resDrmseLoAeComb,\, \resDrmseHiAeComb]$ excluding zero. The heterodyned US channel thus supplies genuinely non-redundant information; the modest absolute size of the gain reflects the relative weakness of the channel (\resNretUs{} retained features, individually weaker than AE's) rather than redundancy."""

# Finding 4: not ALL within-sensor pairs are significant -- the two linear families do not differ.
old4 = r"""    estimates; corrected significance tests confirm that all pairwise model differences within each sensor configuration are significant after Holm--Bonferroni correction, with LightGBM strongest throughout."""
new4 = r"""    estimates; corrected significance tests confirm that LightGBM significantly outperforms both linear families within each sensor configuration (Holm--Bonferroni corrected), while the two linear families are statistically indistinguishable."""

for tag, old in [("f2", old2), ("f4", old4)]:
    if data.count(cv(old)) != 1:
        raise SystemExit(f"{tag} matched {data.count(cv(old))} times")
data = data.replace(cv(old2), cv(new2)).replace(cv(old4), cv(new4))
p.write_bytes(data.encode("utf-8"))
print("OK")
