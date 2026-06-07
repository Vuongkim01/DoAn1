"""
Module huan luyen mo hinh phan loai y dinh (Intent Classification).

Module nay thuc hien:
    1. Doc va tien xu ly du lieu tu dataset.csv
    2. Chia tap du lieu thanh train/test (80/20)
    3. Vector hoa van ban bang TF-IDF
    4. Huan luyen 2 mo hinh: Multinomial Naive Bayes va SVM
    5. Danh gia va so sanh hieu suat
    6. Ve confusion matrix
    7. Luu mo hinh tot nhat va vectorizer
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Su dung backend khong can GUI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Import module tien xu ly
from preprocessing import tien_xu_ly


def doc_du_lieu(duong_dan):
    """
    Doc du lieu tu file CSV.

    Args:
        duong_dan (str): Duong dan toi file CSV.

    Returns:
        pandas.DataFrame: DataFrame chua du lieu da doc.

    Raises:
        FileNotFoundError: Neu file khong ton tai.
    """
    print("[INFO] Dang doc du lieu tu: {}".format(duong_dan))
    df = pd.read_csv(duong_dan)
    print("[OK] Da doc {} mau du lieu".format(len(df)))
    print("[INFO] Phan bo nhan:")
    print(df['intent'].value_counts().to_string())
    print()
    return df


def tien_xu_ly_du_lieu(df):
    """
    Tien xu ly toan bo du lieu trong DataFrame.

    Ap dung ham tien_xu_ly() cho tung cau trong cot 'text'.

    Args:
        df (pandas.DataFrame): DataFrame chua cot 'text' can xu ly.

    Returns:
        pandas.DataFrame: DataFrame da duoc tien xu ly.
    """
    print("[INFO] Dang tien xu ly du lieu...")
    df = df.copy()
    df['text_processed'] = df['text'].apply(tien_xu_ly)
    print("[OK] Tien xu ly hoan tat!")
    print()
    return df


def chia_du_lieu(X, y, ti_le_test=0.2, random_state=42):
    """
    Chia du lieu thanh tap huan luyen va tap kiem tra.

    Args:
        X (pandas.Series): Du lieu dau vao (van ban da xu ly).
        y (pandas.Series): Nhan tuong ung.
        ti_le_test (float): Ty le du lieu kiem tra (mac dinh 0.2 = 20%).
        random_state (int): Seed cho bo sinh so ngau nhien.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ti_le_test, random_state=random_state, stratify=y
    )
    print("[INFO] Chia du lieu: Train={} | Test={}".format(len(X_train), len(X_test)))
    print()
    return X_train, X_test, y_train, y_test


def vector_hoa_tfidf(X_train, X_test):
    """
    Vector hoa van ban su dung TF-IDF.

    TF-IDF (Term Frequency - Inverse Document Frequency) chuyen doi
    van ban thanh vector so dua tren tan suat xuat hien cua tu.

    Args:
        X_train (pandas.Series): Tap huan luyen (van ban).
        X_test (pandas.Series): Tap kiem tra (van ban).

    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, vectorizer)
            - X_train_tfidf: Ma tran TF-IDF cua tap train
            - X_test_tfidf: Ma tran TF-IDF cua tap test
            - vectorizer: Doi tuong TfidfVectorizer da fit
    """
    print("[INFO] Dang vector hoa van ban bang TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,      # So luong tu vung toi da
        ngram_range=(1, 2),     # Su dung unigram va bigram
        sublinear_tf=True       # Ap dung log scaling cho TF
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("[OK] Kich thuoc ma tran TF-IDF: {}".format(X_train_tfidf.shape))
    print("   So dac trung (features): {}".format(len(vectorizer.get_feature_names_out())))
    print()
    return X_train_tfidf, X_test_tfidf, vectorizer


def huan_luyen_naive_bayes(X_train, y_train):
    """
    Huan luyen mo hinh Multinomial Naive Bayes.

    Multinomial NB phu hop cho bai toan phan loai van ban
    voi dac trung TF-IDF (cac gia tri khong am).

    Args:
        X_train: Ma tran dac trung TF-IDF tap huan luyen.
        y_train: Nhan tap huan luyen.

    Returns:
        MultinomialNB: Mo hinh da duoc huan luyen.
    """
    print("[INFO] Dang huan luyen mo hinh Multinomial Naive Bayes...")
    model_nb = MultinomialNB(alpha=1.0)  # alpha: tham so Laplace smoothing
    model_nb.fit(X_train, y_train)
    print("[OK] Huan luyen Naive Bayes hoan tat!")
    return model_nb


def huan_luyen_svm(X_train, y_train):
    """
    Huan luyen mo hinh Support Vector Machine theo chien luoc One-vs-Rest.

    Theo co so ly thuyet (Muc 2.3.1.e), SVC duoc boc trong
    OneVsRestClassifier de giai quyet bai toan phan loai 4 y dinh
    theo chien luoc OvR (thay vi OvO mac dinh cua sklearn).

    GridSearchCV tu dong quet khong gian sieu tham so C trong
    [0.1, 1, 10] ket hop kiem chung cheo K-Fold (cv=5) de tim
    sieu phang phan tach toi uu va ngan chan Overfitting.

    Args:
        X_train: Ma tran dac trung TF-IDF tap huan luyen.
        y_train: Nhan tap huan luyen.

    Returns:
        tuple: (best_model, best_params)
            - best_model: Mo hinh SVM OvR da duoc tinh chinh toi uu.
            - best_params: Dict chua sieu tham so toi uu tim duoc.
    """
    print("[INFO] Dang huan luyen mo hinh SVM (OneVsRest + GridSearchCV)...")
    print("   Chien luoc: One-vs-Rest (OvR)")
    print("   Khong gian tim kiem: C in [0.1, 1, 10]")
    print("   Kiem chung cheo: K-Fold (k=5)")

    # Boc SVC trong OneVsRestClassifier theo dung chien luoc OvR
    ovr_svm = OneVsRestClassifier(SVC(kernel='linear', random_state=42))

    # Dinh nghia khong gian sieu tham so can quet
    # Tham so C dieu chinh muc do chap nhan loi phan loai
    param_grid = {
        'estimator__C': [0.1, 1, 10]  # Quet C theo bao cao muc 3.6
    }

    # GridSearchCV tu dong tim ket hop toi uu qua K-Fold Cross Validation
    grid_search = GridSearchCV(
        ovr_svm,
        param_grid,
        cv=5,               # K-Fold voi k=5
        scoring='accuracy', # Tieu chi danh gia
        n_jobs=-1,          # Dung tat ca CPU cores
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    best_idx = grid_search.best_index_
    cv_mean = grid_search.cv_results_['mean_test_score'][best_idx]
    cv_std = grid_search.cv_results_['std_test_score'][best_idx]
    print("[K-FOLD] Accuracy trung binh: {:.2f}% \u00b1 {:.2f}%".format(cv_mean * 100, cv_std * 100))

    print("[OK] Huan luyen SVM hoan tat!")
    print("   Sieu tham so toi uu: {}".format(best_params))
    print("   CV Accuracy tot nhat: {:.4f}".format(grid_search.best_score_))
    return best_model, best_params


def danh_gia_mo_hinh(model, X_test, y_test, ten_mo_hinh):
    """
    Danh gia mo hinh va in ket qua chi tiet.

    Tinh toan cac chi so:
        - Accuracy: Do chinh xac tong the
        - Precision: Ty le du doan dung trong cac du doan positive
        - Recall: Ty le phat hien dung cac mau positive
        - F1-score: Trung binh dieu hoa cua Precision va Recall

    Args:
        model: Mo hinh da huan luyen.
        X_test: Ma tran dac trung tap kiem tra.
        y_test: Nhan thuc te tap kiem tra.
        ten_mo_hinh (str): Ten mo hinh de hien thi.

    Returns:
        dict: Dictionary chua cac chi so danh gia.
    """
    # Du doan tren tap test
    y_pred = model.predict(X_test)

    # Tinh cac chi so danh gia (weighted average cho multi-class)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # In ket qua
    print("\n" + "=" * 60)
    print("KET QUA DANH GIA: {}".format(ten_mo_hinh))
    print("=" * 60)
    print("  Accuracy  : {:.4f} ({:.2f}%)".format(accuracy, accuracy * 100))
    print("  Precision : {:.4f} ({:.2f}%)".format(precision, precision * 100))
    print("  Recall    : {:.4f} ({:.2f}%)".format(recall, recall * 100))
    print("  F1-Score  : {:.4f} ({:.2f}%)".format(f1, f1 * 100))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        'Model': ten_mo_hinh,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4)
    }


def ve_confusion_matrix(model, X_test, y_test, ten_mo_hinh, duong_dan_luu, labels):
    """
    Ve va luu confusion matrix duoi dang heatmap.

    Confusion matrix giup truc quan hoa ket qua phan loai,
    cho thay mo hinh phan loai dung/sai o nhung lop nao.

    Args:
        model: Mo hinh da huan luyen.
        X_test: Ma tran dac trung tap kiem tra.
        y_test: Nhan thuc te.
        ten_mo_hinh (str): Ten mo hinh.
        duong_dan_luu (str): Duong dan luu file anh.
        labels (list): Danh sach nhan cac lop.
    """
    # Du doan
    y_pred = model.predict(X_test)

    # Tinh confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Ve heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,           # Hien thi so tren o
        fmt='d',              # Dinh dang so nguyen
        cmap='Blues',          # Bang mau
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Confusion Matrix - {}'.format(ten_mo_hinh), fontsize=14, fontweight='bold')
    plt.xlabel('Nhan du doan (Predicted)', fontsize=12)
    plt.ylabel('Nhan thuc te (Actual)', fontsize=12)
    plt.tight_layout()

    # Luu file anh
    plt.savefig(duong_dan_luu, dpi=150, bbox_inches='tight')
    plt.close()
    print("[SAVE] Da luu confusion matrix: {}".format(duong_dan_luu))


def so_sanh_mo_hinh(ket_qua_nb, ket_qua_svm, duong_dan_luu):
    """
    So sanh ket qua cua 2 mo hinh va luu thanh file CSV.

    Tao bang so sanh gom: Model, Accuracy, Precision, Recall, F1
    va in ra man hinh duoi dang bang.

    Args:
        ket_qua_nb (dict): Ket qua danh gia Naive Bayes.
        ket_qua_svm (dict): Ket qua danh gia SVM.
        duong_dan_luu (str): Duong dan luu file CSV.

    Returns:
        pandas.DataFrame: Bang so sanh 2 mo hinh.
    """
    # Tao DataFrame so sanh
    df_so_sanh = pd.DataFrame([ket_qua_nb, ket_qua_svm])

    # In bang so sanh
    print("\n" + "=" * 60)
    print("BANG SO SANH 2 MO HINH")
    print("=" * 60)
    print(df_so_sanh.to_string(index=False))
    print()

    # Luu file CSV
    df_so_sanh.to_csv(duong_dan_luu, index=False, encoding='utf-8-sig')
    print("[SAVE] Da luu bang so sanh: {}".format(duong_dan_luu))

    return df_so_sanh


def luu_mo_hinh(model_nb, model_svm, ket_qua_nb, ket_qua_svm, vectorizer):
    """
    Luu mo hinh tot nhat va vectorizer.

    So sanh Accuracy cua 2 mo hinh:
        - Neu SVM cao hon -> luu SVM
        - Nguoc lai -> luu Naive Bayes

    Args:
        model_nb: Mo hinh Naive Bayes.
        model_svm: Mo hinh SVM.
        ket_qua_nb (dict): Ket qua danh gia NB.
        ket_qua_svm (dict): Ket qua danh gia SVM.
        vectorizer: Doi tuong TfidfVectorizer.
    """
    # Lay duong dan thu muc hien tai
    thu_muc = os.path.dirname(os.path.abspath(__file__))

    # So sanh Accuracy va chon mo hinh tot nhat
    if ket_qua_svm['Accuracy'] >= ket_qua_nb['Accuracy']:
        mo_hinh_tot_nhat = model_svm
        ten_tot_nhat = "SVM"
    else:
        mo_hinh_tot_nhat = model_nb
        ten_tot_nhat = "Multinomial Naive Bayes"

    # Luu mo hinh
    duong_dan_model = os.path.join(thu_muc, 'model.pkl')
    joblib.dump(mo_hinh_tot_nhat, duong_dan_model)
    print("\n[BEST] Mo hinh tot nhat: {}".format(ten_tot_nhat))
    print("[SAVE] Da luu mo hinh: {}".format(duong_dan_model))

    # Luu vectorizer
    duong_dan_vectorizer = os.path.join(thu_muc, 'vectorizer.pkl')
    joblib.dump(vectorizer, duong_dan_vectorizer)
    print("[SAVE] Da luu vectorizer: {}".format(duong_dan_vectorizer))


def main():
    """
    Ham chinh - chay toan bo pipeline huan luyen.

    Pipeline:
        1. Doc du lieu
        2. Tien xu ly
        3. Chia train/test
        4. Vector hoa TF-IDF
        5. Huan luyen Naive Bayes
        6. Huan luyen SVM
        7. Danh gia ca 2 mo hinh
        8. Ve confusion matrix
        9. So sanh va luu ket qua
        10. Luu mo hinh tot nhat
    """
    print("=" * 60)
    print("HE THONG PHAN LOAI Y DINH NGUOI DUNG")
    print("   Chatbot Tu Van May Tinh")
    print("   Su dung TF-IDF + SVM + Multinomial Naive Bayes")
    print("=" * 60)
    print()

    # ============================
    # 1. Doc du lieu
    # ============================
    thu_muc = os.path.dirname(os.path.abspath(__file__))
    duong_dan_csv = os.path.join(thu_muc, 'dataset.csv')
    df = doc_du_lieu(duong_dan_csv)

    # ============================
    # 2. Tien xu ly du lieu
    # ============================
    df = tien_xu_ly_du_lieu(df)

    # ============================
    # 3. Chia tap train/test (80/20)
    # ============================
    X = df['text_processed']
    y = df['intent']
    X_train, X_test, y_train, y_test = chia_du_lieu(X, y, ti_le_test=0.2)

    # ============================
    # 4. Vector hoa TF-IDF
    # ============================
    X_train_tfidf, X_test_tfidf, vectorizer = vector_hoa_tfidf(X_train, X_test)

    # ============================
    # 5. Huan luyen Multinomial Naive Bayes
    # ============================
    model_nb = huan_luyen_naive_bayes(X_train_tfidf, y_train)

    # ============================
    # 6. Huan luyen SVM (OneVsRest + GridSearchCV)
    # ============================
    model_svm, best_params = huan_luyen_svm(X_train_tfidf, y_train)
    print("[INFO] Sieu tham so SVM toi uu da chon: {}".format(best_params))

    # ============================
    # 7. Danh gia mo hinh
    # ============================
    ket_qua_nb = danh_gia_mo_hinh(model_nb, X_test_tfidf, y_test, "Multinomial Naive Bayes")
    ket_qua_svm = danh_gia_mo_hinh(model_svm, X_test_tfidf, y_test, "SVM OneVsRest (kernel=linear, C toi uu)")

    # ============================
    # 8. Ve Confusion Matrix
    # ============================
    # Tao thu muc results neu chua co
    thu_muc_results = os.path.join(thu_muc, 'results')
    os.makedirs(thu_muc_results, exist_ok=True)

    # Lay danh sach nhan
    labels = sorted(y.unique())

    print("\n[INFO] Dang ve Confusion Matrix...")
    ve_confusion_matrix(
        model_nb, X_test_tfidf, y_test,
        "Multinomial Naive Bayes",
        os.path.join(thu_muc_results, 'confusion_matrix_nb.png'),
        labels
    )
    ve_confusion_matrix(
        model_svm, X_test_tfidf, y_test,
        "SVM OneVsRest (kernel=linear)",
        os.path.join(thu_muc_results, 'confusion_matrix_svm.png'),
        labels
    )

    # ============================
    # 9. So sanh mo hinh
    # ============================
    so_sanh_mo_hinh(
        ket_qua_nb, ket_qua_svm,
        os.path.join(thu_muc_results, 'metrics.csv')
    )

    # ============================
    # 10. Luu mo hinh tot nhat
    # ============================
    luu_mo_hinh(model_nb, model_svm, ket_qua_nb, ket_qua_svm, vectorizer)

    print("\n" + "=" * 60)
    print("HOAN TAT! Da sinh ra cac file:")
    print("   model.pkl          - Mo hinh tot nhat")
    print("   vectorizer.pkl     - TF-IDF Vectorizer")
    print("   results/metrics.csv             - Bang so sanh")
    print("   results/confusion_matrix_nb.png - CM Naive Bayes")
    print("   results/confusion_matrix_svm.png - CM SVM")
    print("=" * 60)


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    main()
