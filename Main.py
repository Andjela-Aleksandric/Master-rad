import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats

# GLOBALNE KONSTANTE
SKALA_MAX = 10.0
MIN_SIRINA_KLASTERA = 0.15 * SKALA_MAX  # Minimalno 15% raspona skale (1.5)

# Konstante za širinu preklopa fuzzy skupova
DELTA = 0.05 * SKALA_MAX  # 5% skale (0.5) za srednje (trougaone) funkcije
DELTA_TRAP = 0.10 * SKALA_MAX  # 10% skale (1.0) za krajnje (trapezoidne) funkcije

# Podešavanje globalnog stila za grafikone
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (10, 7)
})


# 1. UČITAVANJE I STRUKTURIRANJE PODATAKA
df = pd.read_csv("Istraživanje finansijske pismenosti i tolerancije na rizik građana Srbije.csv")
print(f"Ukupno učitanih odgovora iz ankete: {len(df)}")

df.columns = [
    "starosna_grupa", "status", "ima_investicije", "mesecni_iznos", "gde_drzi_novac",
    "cilj_investiranja", "reakcija_pad", "prihvatljiv_pad", "investicioni_horizont",
    "prioritet", "etf_znanje", "tehnicko_znanje", "da_li_bi_koristio_alat"
]

# Skaliranje odgovora na interval [0, 1]
reaction_map = {"Odmah bih prodao/la sve da sprečim dalji gubitak": 0.0, "Ne bih ništa menjao/la": 0.5,
                "Dokupio/la bih još verujem u dugoročni rast": 1.0,
                "Dokupio/la bih još jer verujem u dugoročni rast": 1.0}
drop_map = {"Do 5%": 0.0, "Do 15%": 0.5, "Do 30%": 1.0}
priority_map = {"Stabilnost i manji rizik, čak i uz manji prinos": 0.0, "Balans rizika i prinosa": 0.5,
                "Veći prinos, čak i uz veći rizik": 1.0}
horizon_map = {"< 1 godine": 0.0, "1–3 godine": 0.33, "3–7 godina": 0.66, "7+ godina": 1.0}
technical_map = {"Ne": 0.0, "Delimično": 0.5, "Možda": 0.5, "Da": 1.0}

df["reaction_score"] = df["reakcija_pad"].map(reaction_map).fillna(0.5)
df["drop_score"] = df["prihvatljiv_pad"].map(drop_map).fillna(0.5)
df["priority_score"] = df["prioritet"].map(priority_map).fillna(0.5)
df["horizon_score"] = df["investicioni_horizont"].map(horizon_map).fillna(0.5)
df["etf_score"] = (pd.to_numeric(df["etf_znanje"], errors="coerce").fillna(3) - 1) / 4.0
df["technical_score"] = df["tehnicko_znanje"].map(technical_map).fillna(0.5)

# Detekcija atipičnih odgovora
df["suspicious"] = 0
df.loc[(df["priority_score"] == 1.0) & (df["reaction_score"] == 0.0), "suspicious"] = 1
df.loc[(df["drop_score"] == 1.0) & (df["reaction_score"] == 0.0), "suspicious"] = 1
df.loc[(df["priority_score"] == 1.0) & (df["drop_score"] == 1.0) & (df["etf_score"] == 0.0), "suspicious"] = 1

print(f"Broj sumnjivih/kontradiktornih odgovora: {df['suspicious'].sum()} ({100 * df['suspicious'].mean():.2f}%)")

# 2. IBA
def generalized_boolean_product(a, b):
    return np.minimum(a, b)


def iba_logical_aggregation(a, b, alpha=0.75):

    gp = generalized_boolean_product(a, b)

    # IBA disjunkcija
    iba_or = a + b - gp

    # logička agregacija
    return alpha * gp + (1 - alpha) * iba_or

risk_core = iba_logical_aggregation(
    df["reaction_score"],
    df["drop_score"],
    alpha=0.80
)

df["risk_scaled"] = (
    0.7 * risk_core +
    0.3 * df["priority_score"]
)

df["knowledge_scaled"] = iba_logical_aggregation(
    df["etf_score"],
    df["technical_score"],
    alpha=0.75
)
df["horizon_scaled"] = df["horizon_score"]

# Skaliranje na interval [0, 10] i podela skupa
for col in ["risk_scaled", "knowledge_scaled", "horizon_scaled"]:
    df[col] = ((df[col] - df[col].min()) / (df[col].max() - df[col].min())) * SKALA_MAX

df_clean = df[df["suspicious"] == 0].copy().reset_index(drop=True)
df_suspicious = df[df["suspicious"] == 1].copy().reset_index(drop=True)

# 3. K-MEANS OPTIMIZACIJA SA PROVEROM ŠIRINE KLASTERA
def get_safe_bounds(series, col_name):
    km = KMeans(n_clusters=3, random_state=42, n_init=20)
    data_frame = series.to_frame()
    km.fit(data_frame)
    centers = sorted(km.cluster_centers_.flatten())
    q1 = (centers[0] + centers[1]) / 2
    q2 = (centers[1] + centers[2]) / 2

    if (q2 - q1) < MIN_SIRINA_KLASTERA:
        print(f" Upozorenje: Klasteri za [{col_name}] su previše zbijeni (širina: {q2 - q1:.2f} < {MIN_SIRINA_KLASTERA:.1f}).")
        print(f"            U slučaju nestabilne segmentacije korišćeni su tercili distribucije (33% i 66%).")
        q1 = series.quantile(0.33)
        q2 = series.quantile(0.66)
    else:
        print(f" Uspešna K-Means segmentacija za [{col_name}].")

    return q1, q2

print("\nEVALUACIJA I VERIFIKACIJA GRANICA")
risk_q1, risk_q2 = get_safe_bounds(df["risk_scaled"], "Tolerancija na rizik")
knowledge_q1, knowledge_q2 = get_safe_bounds(df["knowledge_scaled"], "Finansijsko znanje")
horizon_q1, horizon_q2 = get_safe_bounds(df["horizon_scaled"], "Investicioni horizont")

print(f"\nKonačne prelomne tačke modela:\n -> Rizik (RT): [{risk_q1:.2f}, {risk_q2:.2f}]\n -> Znanje (FK): [{knowledge_q1:.2f}, {knowledge_q2:.2f}]\n -> Horizont (IH): [{horizon_q1:.2f}, {horizon_q2:.2f}]")

# 4. MAMDANI FAZI SISTEM
x_universe = np.arange(0, 10.01, 0.01)
risk = ctrl.Antecedent(x_universe, "risk")
knowledge = ctrl.Antecedent(x_universe, "knowledge")
horizon = ctrl.Antecedent(x_universe, "horizon")
profile = ctrl.Consequent(x_universe, "profile")

risk_low_mf = fuzz.trapmf(risk.universe, [0, 0, max(0, risk_q1 - DELTA_TRAP), min(SKALA_MAX, risk_q1 + DELTA_TRAP)])
risk_med_mf = fuzz.trimf(risk.universe, [max(0, risk_q1 - DELTA), (risk_q1 + risk_q2) / 2, min(SKALA_MAX, risk_q2 + DELTA)])
risk_high_mf = fuzz.trapmf(risk.universe, [max(0, risk_q2 - DELTA_TRAP), min(SKALA_MAX, risk_q2 + DELTA_TRAP), SKALA_MAX, SKALA_MAX])

risk["low"] = risk_low_mf
risk["medium"] = risk_med_mf
risk["high"] = risk_high_mf

knowledge_low_mf = fuzz.trapmf(knowledge.universe, [0, 0, max(0, knowledge_q1 - DELTA_TRAP), min(SKALA_MAX, knowledge_q1 + DELTA_TRAP)])
knowledge_med_mf = fuzz.trimf(knowledge.universe, [max(0, knowledge_q1 - DELTA), (knowledge_q1 + knowledge_q2) / 2, min(SKALA_MAX, knowledge_q2 + DELTA)])
knowledge_high_mf = fuzz.trapmf(knowledge.universe, [max(0, knowledge_q2 - DELTA_TRAP), min(SKALA_MAX, knowledge_q2 + DELTA_TRAP), SKALA_MAX, SKALA_MAX])

knowledge["low"] = knowledge_low_mf
knowledge["medium"] = knowledge_med_mf
knowledge["high"] = knowledge_high_mf

horizon_short_mf = fuzz.trapmf(horizon.universe, [0, 0, max(0, horizon_q1 - DELTA_TRAP), min(SKALA_MAX, horizon_q1 + DELTA_TRAP)])
horizon_med_mf = fuzz.trimf(horizon.universe, [max(0, horizon_q1 - DELTA), (horizon_q1 + horizon_q2) / 2, min(SKALA_MAX, horizon_q2 + DELTA)])
horizon_long_mf = fuzz.trapmf(horizon.universe, [max(0, horizon_q2 - DELTA_TRAP), min(SKALA_MAX, horizon_q2 + DELTA_TRAP), SKALA_MAX, SKALA_MAX])

horizon["short"] = horizon_short_mf
horizon["medium"] = horizon_med_mf
horizon["long"] = horizon_long_mf

profile["conservative"] = fuzz.trapmf(profile.universe, [0, 0, 2.0, 4.5])
profile["moderate"] = fuzz.trimf(profile.universe, [3.5, 5.5, 7.5])
profile["aggressive"] = fuzz.trapmf(profile.universe, [6.5, 8.5, SKALA_MAX, SKALA_MAX])

rules = [
    # RISK LOW
    ctrl.Rule(risk["low"] & knowledge["low"] & horizon["short"], profile["conservative"]),
    ctrl.Rule(risk["low"] & knowledge["low"] & horizon["medium"], profile["conservative"]),
    ctrl.Rule(risk["low"] & knowledge["low"] & horizon["long"], profile["conservative"]),
    ctrl.Rule(risk["low"] & knowledge["medium"] & horizon["short"], profile["conservative"]),
    ctrl.Rule(risk["low"] & knowledge["medium"] & horizon["medium"], profile["conservative"]),
    ctrl.Rule(risk["low"] & knowledge["medium"] & horizon["long"], profile["moderate"]),
    ctrl.Rule(risk["low"] & knowledge["high"] & horizon["short"], profile["conservative"]),
    ctrl.Rule(risk["low"] & knowledge["high"] & horizon["medium"], profile["moderate"]),
    ctrl.Rule(risk["low"] & knowledge["high"] & horizon["long"], profile["moderate"]),
    # RISK MEDIUM
    ctrl.Rule(risk["medium"] & knowledge["low"] & horizon["short"], profile["conservative"]),
    ctrl.Rule(risk["medium"] & knowledge["low"] & horizon["medium"], profile["conservative"]),
    ctrl.Rule(risk["medium"] & knowledge["low"] & horizon["long"], profile["moderate"]),
    ctrl.Rule(risk["medium"] & knowledge["medium"] & horizon["short"], profile["conservative"]),
    ctrl.Rule(risk["medium"] & knowledge["medium"] & horizon["medium"], profile["moderate"]),
    ctrl.Rule(risk["medium"] & knowledge["medium"] & horizon["long"], profile["moderate"]),
    ctrl.Rule(risk["medium"] & knowledge["high"] & horizon["short"], profile["moderate"]),
    ctrl.Rule(risk["medium"] & knowledge["high"] & horizon["medium"], profile["moderate"]),
    ctrl.Rule(risk["medium"] & knowledge["high"] & horizon["long"], profile["aggressive"]),
    # RISK HIGH
    ctrl.Rule(risk["high"] & knowledge["low"] & horizon["short"], profile["moderate"]),
    ctrl.Rule(risk["high"] & knowledge["low"] & horizon["medium"], profile["moderate"]),
    ctrl.Rule(risk["high"] & knowledge["low"] & horizon["long"], profile["moderate"]),
    ctrl.Rule(risk["high"] & knowledge["medium"] & horizon["short"], profile["moderate"]),
    ctrl.Rule(risk["high"] & knowledge["medium"] & horizon["medium"], profile["moderate"]),
    ctrl.Rule(risk["high"] & knowledge["medium"] & horizon["long"], profile["aggressive"]),
    ctrl.Rule(risk["high"] & knowledge["high"] & horizon["short"], profile["moderate"]),
    ctrl.Rule(risk["high"] & knowledge["high"] & horizon["medium"], profile["aggressive"]),
    ctrl.Rule(risk["high"] & knowledge["high"] & horizon["long"], profile["aggressive"]),
]

control_system = ctrl.ControlSystem(rules)

# 5. POKRETANJE INFERENCE SIMULACIJE
def run_fuzzy_inference(dataframe):
    computed_scores = []
    for idx, row in dataframe.iterrows():
        fis_sim = ctrl.ControlSystemSimulation(control_system)
        fis_sim.input["risk"] = row["risk_scaled"]
        fis_sim.input["knowledge"] = row["knowledge_scaled"]
        fis_sim.input["horizon"] = row["horizon_scaled"]
        try:
            fis_sim.compute()
            computed_scores.append(fis_sim.output["profile"])
        except Exception as e:
            print(f"Kritički matematički prekid proračuna na indeksu {idx}: {e}")
            raise
    return computed_scores

df["fuzzy_profile_score"] = run_fuzzy_inference(df)

if len(df_suspicious) > 0 and len(df_clean) > 0:
    clean_scores = run_fuzzy_inference(df_clean)
    susp_scores = run_fuzzy_inference(df_suspicious)
    stat, p_value = stats.mannwhitneyu(clean_scores, susp_scores, alternative='two-sided')
    print("\n" + "=" * 70 + "\nPROVEREN MANN-WHITNEY U TEST\n" + "=" * 70)
    print(f"U-statistika: {stat:.4f} | p-vrednost: {p_value:.6f} (p >= 0.05 -> Zadržavanje opravdano)")
    print("=" * 70)

# granice
mu_conservative = fuzz.interp_membership(x_universe, profile["conservative"].mf, x_universe)
mu_moderate = fuzz.interp_membership(x_universe, profile["moderate"].mf, x_universe)
mu_aggressive = fuzz.interp_membership(x_universe, profile["aggressive"].mf, x_universe)

# granica izmedju konzervativnog i umerenog
idx_q1 = np.argmin(np.abs(mu_conservative - mu_moderate) + (x_universe < 2.0) * 10 + (x_universe > 5.5) * 10)
score_q1 = round(x_universe[idx_q1], 2)

# granica izmedju umerenog i agresivnog
idx_q2 = np.argmin(np.abs(mu_moderate - mu_aggressive) + (x_universe < 5.5) * 10 + (x_universe > 8.5) * 10)
score_q2 = round(x_universe[idx_q2], 2)

print("\n" + "=" * 60)
print(f"GRANICE FAZI SKUPOVA:")
print(f" -> Konzervativan / Umeren: {score_q1}")
print(f" -> Umeren / Agresivan: {score_q2}")
print("=" * 60)

# kategorizacija preko granica
df["profile_category"] = np.where(df["fuzzy_profile_score"] <= score_q1, "Konzervativan",
                                  np.where(df["fuzzy_profile_score"] < score_q2, "Umeren", "Agresivan"))


# 6. ISPIS TABELA
print("\n" + "=" * 60 + "\nDISTRIBUCIJA I DESKRIPTIVNA STATISTIKA INVESTICIONIH PROFILA\n" + "=" * 60)
counts = df["profile_category"].value_counts()
pcts = df["profile_category"].value_counts(normalize=True) * 100
means = df.groupby("profile_category")[["risk_scaled", "knowledge_scaled", "horizon_scaled"]].mean()

tabela_4 = pd.DataFrame({"Broj ispitanika (f)": counts, "Procenat (%)": pcts.round(2)}).join(means.round(2))
tabela_4.columns = ["Broj ispitanika (f)", "Procenat (%)", "Prosečan RT", "Prosečan FK", "Prosečan IH"]
print(tabela_4.reindex(["Konzervativan", "Umeren", "Agresivan"]))

print("\n" + "=" * 60 + "\nDESKRIPTIVNA STATISTIKA KONTINUALNOG MAMDANI SKORA (MS*)\n" + "=" * 60)
desc = df["fuzzy_profile_score"].describe()
mapa_statistika = {"count": "Broj uzoraka", "mean": "Srednja vrednost (Mean)", "std": "Standardna devijacija",
                   "min": "Minimum", "25%": "Prvi kvartil (Q1)", "50%": "Medijana", "75%": "Treći kvartil (Q3)",
                   "max": "Maksimum"}
tabela_5 = pd.DataFrame({"Statistički pokazatelj": desc.index, "Vrednost skora (MS*)": desc.values.round(2)})
tabela_5["Statistički pokazatelj"] = tabela_5["Statistički pokazatelj"].map(mapa_statistika)
print(tabela_5)


# GENERISANJE UNKARSNE TABELE (STAROST vs PROFIL)
stvarne_grupe = df["starosna_grupa"].dropna().unique()
redosled_starosnih_grupa = sorted(stvarne_grupe)
redoseled_profila = ["Konzervativan", "Umeren", "Agresivan"]

# Kreiramo unakrsnu tabelu (crosstab) u procentima
tabela_starost_profil = pd.crosstab(
    df["starosna_grupa"],
    df["profile_category"],
    normalize='index'
) * 100

# Reindeksiramo tabelu i menjamo NaN vrednosti u 0
tabela_starost_profil = tabela_starost_profil.reindex(
    index=redosled_starosnih_grupa,
    columns=redoseled_profila
).fillna(0).round(2)

tabela_starost_profil.columns = ["Конзервативан", "Умерен", "Агресиван"]
tabela_starost_profil.index.name = "Старосна група"

print("\n" + "=" * 60 + "\nТабела 4. Расподела инвестиционих профила по старосним групама (%)\n" + "=" * 60)
print(tabela_starost_profil)
print("=" * 60)
print("\n" + "=" * 60 + "\nТабела 5. Просечне вредности индикатора по инвестиционом профилу\n" + "=" * 60)

# Računamo proseke za tri ključna indikatora grupisana po profilima
tabela_indikatora = df.groupby("profile_category")[["risk_scaled", "horizon_scaled", "knowledge_scaled"]].mean()

# Reindeksiramo da redosled profila bude akademski (Konzervativan -> Umeren -> Agresivan)
tabela_indikatora = tabela_indikatora.reindex(["Konzervativan", "Umeren", "Agresivan"])

# Preimenujemo kolone i indeks u skladu sa tvojom slikom iz Word-a
tabela_indikatora.columns = ["Толеранција на ризик", "Инвестициони хоризонт", "Финансијско знање"]
tabela_indikatora.index.name = "Профил"

# Zaokružujemo na 2 decimale i štampamo
print(tabela_indikatora.round(2))
print("=" * 60)

# 7. GRAFIKONI

# Grafik 1: Funkcije pripadnosti
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
axs[0].plot(x_universe, risk_low_mf, label="Ниска", color="#1f77b4")
axs[0].plot(x_universe, risk_med_mf, label="Средња", color="#ff7f0e")
axs[0].plot(x_universe, risk_high_mf, label="Висока", color="#2ca02c")
axs[0].set_title("Функције припадности: Толеранција на ризик (RT)")
axs[0].set_xlabel("Улазне вредности скале")
axs[0].set_ylabel("Степен припадности (μ)")
axs[0].legend(title="Лингвистичке вредности")

axs[1].plot(x_universe, knowledge_low_mf, label="Ниско", color="#1f77b4")
axs[1].plot(x_universe, knowledge_med_mf, label="Средње", color="#ff7f0e")
axs[1].plot(x_universe, knowledge_high_mf, label="Високо", color="#2ca02c")
axs[1].set_title("Функције припадности: Финансијско знање (FK)")
axs[1].set_xlabel("Улазне вредности скале")
axs[1].set_ylabel("Степен припадности (μ)")
axs[1].legend(title="Лингвистичке вредности")

axs[2].plot(x_universe, horizon_short_mf, label="Кратак", color="#1f77b4")
axs[2].plot(x_universe, horizon_med_mf, label="Средњи", color="#ff7f0e")
axs[2].plot(x_universe, horizon_long_mf, label="Дуг", color="#2ca02c")
axs[2].set_title("Функције припадности: Инвестициони хоризонт (IH)")
axs[2].set_xlabel("Улазне вредности скале")
axs[2].set_ylabel("Степен припадности (μ)")
axs[2].legend(title="Лингвистичке вредности")
plt.tight_layout()
plt.savefig("grafik_1_funkcije_pripadnosti.png", dpi=300, bbox_inches='tight')
plt.close()

# Grafik 2: 3D Površina zaključivanja
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
x_mesh, y_mesh = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 10, 20))
z_mesh = np.zeros_like(x_mesh)

for i in range(20):
    for j in range(20):
        sim = ctrl.ControlSystemSimulation(control_system)
        sim.input["risk"] = x_mesh[i, j]
        sim.input["knowledge"] = y_mesh[i, j]
        sim.input["horizon"] = 5.0
        try:
            sim.compute()
            z_mesh[i, j] = sim.output["profile"]
        except Exception as e:
            print(f"Greška tokom formiranja 3D mreže na [{i}, {j}]: {e}")
            raise

surf = ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap="viridis", edgecolor='none', alpha=0.9)
ax.set_xlabel('Толеранција na ризик (RT)')
ax.set_ylabel('Финансијско знање (FK)')
ax.set_zlabel('Мамдани излазни скор (MS*)')
plt.title("Тродимензионална контролна површина ФИС система (Фиксиран Хоризонт = 5.0)")
cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
cbar.set_label('Излазни континуални скор')
plt.tight_layout()
plt.savefig("grafik_2_3d_povrsina.png", dpi=300, bbox_inches='tight')
plt.close()

# SLIKA 17: Kružni dijagram distribucije investicionih profila
plt.figure(figsize=(8, 8))
boje_profila = ["#1f77b4", "#ff7f0e", "#2ca02c"]
kategorije_redom = ["Konzervativan", "Umeren", "Agresivan"]
counts_profil = df["profile_category"].value_counts().reindex(kategorije_redom)
cirilicne_labele = ["Конзервативан", "Умерен", "Агресиван"]

plt.pie(
    counts_profil,
    labels=cirilicne_labele,
    autopct='%1.2f%%',
    startangle=140,
    colors=boje_profila,
    textprops={'fontsize': 12, 'weight': 'bold'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
plt.title("Процентуална дистрибуција крајњих инвестиционих профила испитаника", pad=20, weight='bold')
plt.tight_layout()
plt.savefig("slika_17_distribucija_profila.png", dpi=300, bbox_inches='tight')
plt.close()

# SLIKA 18: Scatter Plot strukture profila
plt.figure(figsize=(10, 7))
mapa_boja_scatter = {"Konzervativan": "#1f77b4", "Umeren": "#ff7f0e", "Agresivan": "#2ca02c"}
mapa_prevoda_za_legendu = {"Konzervativan": "Конзервативан", "Umeren": "Умерен", "Agresivan": "Агресиван"}
kategorije_za_iscrtavanje = ["Konzervativan", "Umeren", "Agresivan"]

for kat in kategorije_za_iscrtavanje:
    sub_df = df[df["profile_category"] == kat]
    plt.scatter(
        sub_df["risk_scaled"],
        sub_df["knowledge_scaled"],
        label=mapa_prevoda_za_legendu[kat],
        color=mapa_boja_scatter[kat],
        alpha=0.75,
        s=60,
        edgecolors='none'
    )
plt.title("Расподела инвестиционих профила у одноsu на толеранцију на ризик i знање", pad=15, weight='bold')
plt.xlabel("Толеранција на ризик (RT)")
plt.ylabel("Финансијско знање (FK)")
plt.xlim(-0.5, 10.5)
plt.ylim(-0.5, 10.5)
plt.legend(title="Инвестициони профили", loc="upper left")
plt.tight_layout()
plt.savefig("slika_18_scatter_profila.png", dpi=300, bbox_inches='tight')
plt.close()

# SLIKA 19: Histogram raspodele kontinualnog Mamdani skora
plt.figure(figsize=(10, 6))
srednja_vrednost = df["fuzzy_profile_score"].mean()
medijana_vrednost = df["fuzzy_profile_score"].median()

sns.histplot(
    df["fuzzy_profile_score"],
    kde=True,
    color="#1f77b4",
    bins=25,
    edgecolor="white",
    line_kws={"linewidth": 2.5, "color": "#d62728"}
)
plt.axvline(srednja_vrednost, color="#d62728", linestyle="--", linewidth=2, label=f"Средња вредност ({srednja_vrednost:.2f})")
plt.axvline(medijana_vrednost, color="#2ca02c", linestyle="-.", linewidth=2, label=f"Медијана ({medijana_vrednost:.2f})")

plt.axvline(score_q1, color="black", linestyle=":", alpha=0.5)
plt.axvline(score_q2, color="black", linestyle=":", alpha=0.5)
plt.text(score_q1 - 0.7, plt.gca().get_ylim()[1]*0.85, "Конзервативан", rotation=90, alpha=0.7, weight='bold')
plt.text((score_q1 + score_q2)/2 - 0.3, plt.gca().get_ylim()[1]*0.85, "Умерен", alpha=0.7, weight='bold')
plt.text(score_q2 + 0.3, plt.gca().get_ylim()[1]*0.85, "Агресиван", rotation=90, alpha=0.7, weight='bold')

plt.title("Емпиријска расподела вредности излазног континуалног Мамдани скора", pad=15, weight='bold')
plt.xlabel("Мамдани излазни скор (MS*)")
plt.ylabel("Фреквенција (број испитаника)")
plt.xlim(0, 10)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("slika_19_histogram_mamdani.png", dpi=300, bbox_inches='tight')
plt.close()

# SLIKA 20: Heatmap korelacione matrice
plt.figure(figsize=(9, 7))
korelaciona_matrica = df[["risk_scaled", "knowledge_scaled", "horizon_scaled", "fuzzy_profile_score"]].corr(method="pearson")
akademski_nazivi_cirilica = ["Толеранција на ризик (RT)", "Финансијско знање (FK)", "Инвестициони хоризонт (IH)", "Мамдани излазни скор (MS*)"]
korelaciona_matrica.columns = akademski_nazivi_cirilica
korelaciona_matrica.index = akademski_nazivi_cirilica

sns.heatmap(
    korelaciona_matrica,
    annot=True,
    cmap="RdBu_r",
    fmt=".3f",
    vmin=-1,
    vmax=1,
    linewidths=1,
    cbar_kws={"label": "Пирсонов коефицијент корелације (r)"},
    annot_kws={"size": 11, "weight": "bold"}
)
plt.title("Корелациона матрица индикатора понашања и резултујућег Мамдани скора", pad=20, weight='bold')
plt.xticks(rotation=15, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("slika_20_korelaciona_matrica.png", dpi=300, bbox_inches='tight')
plt.close()