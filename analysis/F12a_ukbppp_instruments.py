"""cis-pQTL inputs for Mendelian randomisation from the UKB-PPP paper (Sun et al., Nature 2023, Supplementary Information
workbook, saved as data/mr/ukbppp/Sun2023_SupplementaryTables.xlsx).
Writes the rows for the intestinal epithelial proteins (GUCA2A, FABP6, DDC, FABP2, CDH17) of
  Supplementary Table 16 (independent pQTL signals, SuSiE)  -> data/mr/ukbppp/ST16_selected_proteins.csv
  Supplementary Table 9 (significant pQTLs, discovery)      -> data/mr/ukbppp/ST9_selected_proteins.csv
"""
import pandas as pd

from common import PROJ

MR = PROJ / "data" / "mr" / "ukbppp"
PROTS = ["GUCA2A", "FABP6", "DDC", "FABP2", "CDH17"]
XLSX = MR / "Sun2023_SupplementaryTables.xlsx"

# ST16 has a two-row header: group label (row 4) and column label (row 5)
h = pd.read_excel(XLSX, sheet_name="ST16", header=None, skiprows=3, nrows=2)
top = h.iloc[0].ffill()
cols = [str(top[j]) if pd.isna(h.iloc[1, j]) else f"{top[j]} | {h.iloc[1, j]}" for j in range(h.shape[1])]
st16 = pd.read_excel(XLSX, sheet_name="ST16", header=None, skiprows=5)
st16.columns = cols
st16 = st16[st16.iloc[:, 0].astype(str).str.split(":").str[0].isin(PROTS)]
st16.to_csv(MR / "ST16_selected_proteins.csv", index=False)

st9 = pd.read_excel(XLSX, sheet_name="ST9", header=4)
st9 = st9[st9["UKBPPP ProteinID"].astype(str).str.split(":").str[0].isin(PROTS)]
st9.to_csv(MR / "ST9_selected_proteins.csv", index=False)
print("ST16 rows:", len(st16), "ST9 rows:", len(st9))
