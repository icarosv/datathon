import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import trange
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Caminhos dos arquivos processados
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

def load_processed_data():
    """
    Carrega os CSVs processados e retorna três DataFrames:
    prospects_clear, applicants_clear, vagas_clear.
    """
    # Prospects
    prospects_clear = pd.read_csv(DATA_PROCESSED / "prospects.csv", sep=';')

    # Applicants (index = id)
    applicants_clear = pd.read_csv(DATA_PROCESSED / "applicants.csv", sep=';')
    if "id" in applicants_clear.columns:
        applicants_clear.set_index("id", inplace=True)

    # Vagas (index = id)
    vagas_clear = pd.read_csv(DATA_PROCESSED / "vagas.csv", sep=';')
    if "id" in vagas_clear.columns:
        vagas_clear.set_index("id", inplace=True)

    return prospects_clear, applicants_clear, vagas_clear

def build_model_dataset(prospects_clear: pd.DataFrame,
                        applicants_clear: pd.DataFrame,
                        vagas_clear: pd.DataFrame) -> pd.DataFrame:
    """
    Combina os DataFrames prospects, applicants e vagas para formar o dataset final do modelo.
    - prospects_clear: contém id_applicant e id_vaga
    - applicants_clear: indexado por id do aplicante
    - vagas_clear: indexado por id da vaga
    """
    # Merge prospects com applicants (id_applicant ↔ index)
    merged_df = pd.merge(
        prospects_clear,
        applicants_clear,
        left_on="id_applicant",
        right_index=True,
        how="left"
    )

    # Merge do resultado com vagas (id_vaga ↔ index)
    final_df = pd.merge(
        merged_df,
        vagas_clear,
        left_on="id_vaga",
        right_index=True,
        how="left"
    )

    return final_df

def embed_text_columns(
    df: pd.DataFrame,
    text_cols,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 1024,
    device: str | None = None,
    flatten: bool = False
) -> pd.DataFrame:
    """
    Adiciona embeddings de colunas de texto ao próprio DataFrame.

    Params
    ------
    df : DataFrame original (modificado in place e também retornado)
    text_cols : lista com nomes das colunas de texto
    model_name : modelo Sentence-Transformers
    batch_size : tamanho do lote para inferência
    device : 'cpu', 'cuda', 'cuda:0', etc. (auto se None)
    flatten : se True, cria várias colunas float (col_emb_0...); senão, guarda lista/np.array em uma única coluna

    Return
    ------
    df com novas colunas *_emb (ou *_emb_0, *_emb_1...)
    """
    model = SentenceTransformer(model_name, device=device)

    for col in text_cols:
        if col not in df.columns:
            raise KeyError(f"Coluna '{col}' não existe no DataFrame.")

        # Prepara o texto
        texts = df[col].fillna("").astype(str).tolist()

        # Inferência em lotes
        embs = []
        for i in trange(0, len(texts), batch_size, desc=f"Encoding {col}"):
            batch = texts[i:i+batch_size]
            embs.append(
                model.encode(batch, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
            )
        embs = np.vstack(embs)

        if flatten:
            dim = embs.shape[1]
            new_cols = {f"{col}_emb_{i}": embs[:, i] for i in range(dim)}
            df = df.join(pd.DataFrame(new_cols, index=df.index))
        else:
            df[f"{col}_emb"] = list(embs)

    df = df.drop(columns=text_cols)
    return df

if __name__ == "__main__":
    output_file = DATA_PROCESSED / "dataset_for_model.csv"

    if output_file.exists():
        print(f"Arquivo {output_file} encontrado. Pulando a geração do dataset...")
        # Carregar dataset diretamente
        final_df = pd.read_csv(output_file, sep=';')
    else:
        print("Dataset não encontrado. Executando pipeline para gerar dataset_for_model.csv...")
        # 1. Carregar dados
        prospects_clear, applicants_clear, vagas_clear = load_processed_data()

        # 2. Gerar dataset unificado
        final_df = build_model_dataset(prospects_clear, applicants_clear, vagas_clear)
        
    model_file = PROJECT_ROOT / "models" / "model.pkl"
    label_file = PROJECT_ROOT / "models" / "label_encoder.pkl"
    if model_file.exists() and label_file.exists():
        print(f"Modelo já encontrado em {model_file}. Pulando etapa de treino.")
        exit(0)

        # 3. Configurar dispositivo
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # 4. Aplicar embeddings nas colunas desejadas
        text_columns = ["informacoes.texto_x", "informacoes.texto_y"]
        final_df = embed_text_columns(
            final_df,
            text_columns,
            device=device,
            flatten=True
        )
    
        # 5. Salvar o dataset final para futuras etapas de treino/avaliação
        output_file = DATA_PROCESSED / "dataset_for_model.csv"
        final_df.to_csv(output_file, index=True, sep=';')
        print(f"DataFrame final salvo em: {output_file}")
    
    # 6. Separar features (X) e target (y)
    target_column = "classificacao"
    drop_columns = ["id_applicant", "id_vaga", target_column]

    X = final_df.drop(columns=drop_columns)
    y = final_df[target_column]

    print(f"Features e target separados: X={X.shape}, y={y.shape}")

    # 8. Converter colunas booleanas (bool ou object que representam bool) em inteiros
    boolean_like_cols = X.select_dtypes(include=["bool", "object"]).columns
    for col in boolean_like_cols:
        try:
            X[col] = X[col].astype(bool).astype(int)
            print(f"Converted column '{col}' to int.")
        except ValueError:
            print(f"Column '{col}' could not be converted to boolean/int, skipping.")

    # 9. Codificar a variável alvo (classificacao)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Target variable 'classificacao' encoded to numerical values.")
    print("Mapping of original classes to encoded values:")
    for original_class, encoded_value in zip(label_encoder.classes_,
                                             label_encoder.transform(label_encoder.classes_)):
        print(f"  {original_class}: {encoded_value}")

    # 10. Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42
    )

    print(f"Data split: Train={len(X_train)}, Test={len(X_test)}")

    # 11. Imputar valores ausentes, se houver
    if X_train.isnull().values.any() or X_test.isnull().values.any():
        print("Missing values found. Applying imputation with -1.")
        imputer = SimpleImputer(strategy="constant", fill_value=-1)
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
    else:
        print("No missing values found in training or testing data.")

    # 12. Aplicar SMOTE para balancear classes no conjunto de treino
    print("Applying SMOTE to the training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("Oversampling with SMOTE completed.")
    print(f"Original training set shape: {X_train.shape}")
    print(f"Resampled training set shape: {X_train_resampled.shape}")

    # 13. Treinar modelo com RandomizedSearchCV (GPU se disponível)
    print("Starting RandomizedSearchCV...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_resampled, y_train_resampled,
        test_size=0.1, stratify=y_train_resampled, random_state=42
    )

#    param_dist = {
#        "learning_rate": [0.05, 0.1, 0.5],
#        "max_depth": [6, 8, 10],
#        "n_estimators": [200, 300, 500],
#    }

    param_dist = {
        "learning_rate": [0.1],
        "max_depth": [10],
        "n_estimators": [500],
    }

    def run_random_search(tree_method):
        model = XGBClassifier(
            objective="multi:softmax",
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            tree_method=tree_method,
            n_jobs=-1,
        )
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=50,
            cv=3,
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=2,
            random_state=42,
            return_train_score=True,
        )
        search.fit(X_tr, y_tr)
        return search

    try:
        temp_model = XGBClassifier(tree_method="gpu_hist")
        temp_model.fit(X_tr[:10], y_tr[:10])
        print("GPU support detected. Running with GPU.")
        random_search = run_random_search("gpu_hist")
    except Exception as e:
        print(f"GPU failure or lack of support: {e}")
        print("Falling back to CPU...")
        random_search = run_random_search("hist")

    results = pd.DataFrame(random_search.cv_results_)
    print(results[["params", "mean_test_score", "rank_test_score"]].head())

    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best cross-validation F1-weighted score: {random_search.best_score_:.4f}")

    best_model = random_search.best_estimator_

    # 14. Avaliação final do modelo no conjunto de teste
    print("Model Evaluation on Test Set:")

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("\nConfusion Matrix:")
    print(cm)
    
    # Salvar métricas em JSON
    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }


    # 15. Salvar modelo e label_encoder
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    from joblib import dump
    dump(best_model, MODELS_DIR / "model.pkl")
    dump(label_encoder, MODELS_DIR / "label_encoder.pkl")
    print(f"Modelo e LabelEncoder salvos em {MODELS_DIR}")
    metrics_file = MODELS_DIR / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"Métricas salvas em {metrics_file}")

    print("Pipeline concluído com sucesso.")

