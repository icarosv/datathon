#!/usr/bin/env python3
"""
process_data.py

Script de pré-processamento de dados para dashboard de candidatos:

1. Extrai arquivos ZIP de `raw_dir` para `work_dir`.
2. Achata JSONs em DataFrames planos.
3. Aplica transformações em cada DataFrame:
   - vagas → vagas_clean
   - applicants → applicants_clean
   - prospects → prospects_clean
4. Mescla prospects, applicants e vagas em `features.csv`.
5. Salva CSVs limpos em `output_dir`.
"""

import argparse
import json
import logging
import zipfile
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def extract_zips(raw_dir: Path, work_dir: Path):
    """Extrai todos os arquivos .zip de raw_dir para work_dir."""
    work_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in raw_dir.glob("*.zip"):
        logging.info(f"Extraindo {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(work_dir)


def flatten_jsons(work_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Percorre work_dir, carrega cada JSON e achata em DataFrame.
    Retorna dict com chave = nome base do arquivo, valor = df.
    """
    dfs: Dict[str, pd.DataFrame] = {}
    for path in work_dir.rglob("*.json"):
        name = path.stem  # ex: 'vagas', 'applicants', 'prospects'
        logging.info(f"Processando JSON plano: {path.name}")
        data = json.loads(path.read_text(encoding="utf-8"))
        # Normalização
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            # dict-of-dicts: transforma em lista com coluna 'id'
            records = [{"id": k, **v} for k, v in data.items()]
            df = pd.json_normalize(records)
        elif isinstance(data, dict):
            df = pd.json_normalize(data)
        else:
            logging.warning(f"Formato não suportado em {path}, pulando.")
            continue
        dfs[name] = df
        logging.info(f" ‣ '{name}' → {df.shape[0]}×{df.shape[1]}")
    return dfs

def save_flat_jsons(dfs: Dict[str, pd.DataFrame], flat_dir: Path):
    """
    Salva os DataFrames brutos (flat) no diretório especificado.
    """
    flat_dir.mkdir(parents=True, exist_ok=True)
    for name, df in dfs.items():
        out_path = flat_dir / f"{name}_flat.csv"
        df.to_csv(out_path, sep=";", index=False)
        logging.info(f"Flat salvo: {out_path} ({df.shape[0]}x{df.shape[1]})")


def process_prospects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expande a coluna 'prospects' (lista de dicts) em linhas individuais.
    Cria colunas: 'vaga_id', 'id_applicant' e campos adicionais.
    """
    records = []
    for _, row in df.iterrows():
        vaga_id = row.get("id")
        for p in row.get("prospects", []):
            rec = {"vaga_id": vaga_id}
            # detecta chave de applicant
            rec["id_applicant"] = (
                p.get("applicantId") or p.get("applicant_id") or p.get("id")
            )
            # copia demais campos
            for k, v in p.items():
                if k not in ("applicantId", "applicant_id", "id"):
                    rec[k] = v
            records.append(rec)
    out = pd.DataFrame(records)
    logging.info(f"Prospects processados → {out.shape[0]}×{out.shape[1]}")
    return out


def process_vagas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exemplos de transformações em vagas:
    - drop colunas irrelevantes
    - binarização, one-hot e concatenação de texto
    """
    df = df.copy().set_index("id", drop=True)
    # remova colunas desnecessárias
    drop_cols = [
        # coloque aqui as colunas que não interessam
        "informacoes_basicas.data_inicial",
        "informacoes_basicas.data_final",
        # ...
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # transforma 'vaga_sap' em 0/1
    sap = "informacoes_basicas.vaga_sap"
    if sap in df:
        df[sap] = df[sap].map({"Sim": 1, "Não": 0}).fillna(0).astype(int)

    # exemplo: concatenação de texto em única coluna
    text_cols = {
        "informacoes_basicas.titulo_vaga": "titulo vaga",
        # adicione mais se desejar...
    }
    parts = []
    for col, prefix in text_cols.items():
        if col in df:
            parts.append(df[col].fillna("").apply(lambda x: f"{prefix} {x}" if x else ""))
            df.drop(columns=[col], inplace=True)
    if parts:
        df["informacoes_texto"] = pd.concat(parts, axis=1).agg(" ".join, axis=1)

    # one-hot encoding de um campo categórico
    obj = "informacoes_basicas.objetivo_vaga"
    if obj in df:
        df = pd.concat([df, pd.get_dummies(df[obj], prefix="objetivo")], axis=1)
        df.drop(columns=[obj], inplace=True)

    # MultiLabelBinarizer em 'tipo_contratacao'
    tc = "informacoes_basicas.tipo_contratacao"
    if tc in df:
        mlb = MultiLabelBinarizer()
        listas = df[tc].fillna("").str.split(",").tolist()
        dummies = pd.DataFrame(
            mlb.fit_transform(listas),
            index=df.index,
            columns=[f"tipo_{c}" for c in mlb.classes_],
        )
        df = pd.concat([df, dummies], axis=1).drop(columns=[tc])

    logging.info(f"Vagas processadas → {df.shape[0]}×{df.shape[1]}")
    return df


def process_applicants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exemplos de transformações em applicants:
    - drop colunas irrelevantes
    - concatenação de texto
    - binarização de campos
    """
    df = df.copy().set_index("id", drop=True)
    # remova colunas desnecessárias
    drop_cols = [
        "informacoes_pessoais.cpf",
        # ...
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # concatena texto de várias colunas
    text_cols = {
        "cv_pt": "cv pt",
        "infos_basicas.objetivo_profissional": "objetivo prof",
        # ...
    }
    parts = []
    for col, prefix in text_cols.items():
        if col in df:
            parts.append(df[col].fillna("").apply(lambda x: f"{prefix} {x}" if x else ""))
            df.drop(columns=[col], inplace=True)
    if parts:
        df["informacoes_texto"] = pd.concat(parts, axis=1).agg(" ".join, axis=1)

    # exemplo binário
    pcd = "informacoes_pessoais.pcd"
    if pcd in df:
        df[pcd] = df[pcd].map({"Sim": 1, "Não": 0}).fillna(0).astype(int)

    # ordinal mapping em 'nivel_profissional'
    nivel = "informacoes_profissionais.nivel_profissional"
    if nivel in df:
        map_nivel = {
            "Aprendiz": 0,
            "Auxiliar": 1,
            # ...
        }
        df["nivel_profissional_ordinal"] = df[nivel].map(map_nivel).fillna(-1).astype(int)
        df.drop(columns=[nivel], inplace=True)

    logging.info(f"Applicants processados → {df.shape[0]}×{df.shape[1]}")
    return df


def merge_and_save(
    prospects: pd.DataFrame,
    applicants: pd.DataFrame,
    vagas: pd.DataFrame,
    output_dir: Path,
):
    """Faz merge de prospects ↔ applicants ↔ vagas e salva CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # salva intermediários
    applicants.to_csv(output_dir / "applicants.csv", sep=";", index=True)
    vagas.to_csv(output_dir / "vagas.csv", sep=";", index=True)
    prospects.to_csv(output_dir / "prospects.csv", sep=";", index=False)

    # merge final
    merged = (
        prospects.merge(applicants, left_on="id_applicant", right_index=True, how="left")
        .merge(vagas, left_on="vaga_id", right_index=True, how="left")
    )
    merged.to_csv(output_dir / "features.csv", sep=";", index=False)
    logging.info(f"Features mescladas e salvas → {merged.shape[0]}×{merged.shape[1]}")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Pipeline de pré-processamento")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        required=True,
        help="Diretório com arquivos ZIP originais",
    )
    parser.add_argument(
        "--work_dir",
        type=Path,
        required=True,
        help="Diretório temporário para extração e JSONs",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Diretório para salvar CSVs processados",
    )
    parser.add_argument(
    "--flat_dir",
    type=Path,
    required=False,
    default=None,
    help="Diretório para salvar os CSVs flat (sem processamento)"
    )
    args = parser.parse_args()

    # Cria os diretórios se não existirem
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    extract_zips(args.raw_dir, args.work_dir)
    flat = flatten_jsons(args.work_dir)
    
    if args.flat_dir:
        save_flat_jsons(flat, args.flat_dir)

    vagas_df = process_vagas(flat.get("vagas", pd.DataFrame()))
    applicants_df = process_applicants(flat.get("applicants", pd.DataFrame()))
    prospects_df = process_prospects(flat.get("prospects", pd.DataFrame()))

    merge_and_save(prospects_df, applicants_df, vagas_df, args.output_dir)

if __name__ == "__main__":
    main()
