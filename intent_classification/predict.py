"""
Module du doan y dinh (intent) tu cau nhap cua nguoi dung.

Module nay thuc hien:
    1. Load mo hinh da huan luyen (model.pkl)
    2. Load vectorizer (vectorizer.pkl)
    3. Tien xu ly cau nhap
    4. Vector hoa va du doan intent

Su dung:
    python predict.py
"""

import os
import joblib
from preprocessing import tien_xu_ly


def tai_mo_hinh():
    """
    Tai mo hinh va vectorizer da duoc luu.

    Returns:
        tuple: (model, vectorizer)
            - model: Mo hinh phan loai da huan luyen
            - vectorizer: TfidfVectorizer da fit tren tap train

    Raises:
        FileNotFoundError: Neu file model.pkl hoac vectorizer.pkl khong ton tai.
    """
    thu_muc = os.path.dirname(os.path.abspath(__file__))

    duong_dan_model = os.path.join(thu_muc, 'model.pkl')
    duong_dan_vectorizer = os.path.join(thu_muc, 'vectorizer.pkl')

    # Kiem tra file ton tai
    if not os.path.exists(duong_dan_model):
        raise FileNotFoundError(
            "[LOI] Khong tim thay file mo hinh: {}\n"
            "   Hay chay 'python train.py' truoc de huan luyen mo hinh.".format(duong_dan_model)
        )
    if not os.path.exists(duong_dan_vectorizer):
        raise FileNotFoundError(
            "[LOI] Khong tim thay file vectorizer: {}\n"
            "   Hay chay 'python train.py' truoc de huan luyen mo hinh.".format(duong_dan_vectorizer)
        )

    # Load mo hinh va vectorizer
    model = joblib.load(duong_dan_model)
    vectorizer = joblib.load(duong_dan_vectorizer)

    return model, vectorizer


def du_doan_intent(cau_nhap, model, vectorizer):
    """
    Du doan intent tu cau nhap cua nguoi dung.

    Pipeline du doan:
        1. Tien xu ly cau nhap (lowercase, loai ky tu dac biet, tach tu, loai stopwords)
        2. Vector hoa bang TF-IDF vectorizer da fit
        3. Du doan nhan intent bang mo hinh da huan luyen

    Args:
        cau_nhap (str): Cau tieng Viet do nguoi dung nhap.
        model: Mo hinh phan loai da huan luyen.
        vectorizer: TfidfVectorizer da fit.

    Returns:
        str: Nhan intent duoc du doan (Tu_Van, Hoi_Gia, Thong_So, So_Sanh).
    """
    # Buoc 1: Tien xu ly cau nhap
    cau_da_xu_ly = tien_xu_ly(cau_nhap)

    # Buoc 2: Vector hoa bang TF-IDF
    cau_vector = vectorizer.transform([cau_da_xu_ly])

    # Buoc 3: Du doan intent
    intent = model.predict(cau_vector)[0]

    return intent


# Tu dien mo ta y nghia cac intent
MO_TA_INTENT = {
    "Tu_Van": "[Tu van mua hang] - Nguoi dung can duoc tu van chon san pham phu hop",
    "Hoi_Gia": "[Hoi gia san pham] - Nguoi dung muon biet gia cua san pham cu the",
    "Thong_So": "[Hoi thong so ky thuat] - Nguoi dung muon biet cau hinh, thong so san pham",
    "So_Sanh": "[So sanh san pham] - Nguoi dung muon so sanh giua cac san pham"
}


def main():
    """
    Ham chinh - chuong trinh du doan intent tuong tac.

    Cho phep nguoi dung:
        1. Nhap cau hoi tieng Viet
        2. Xem ket qua du doan intent
        3. Tiep tuc nhap hoac thoat
    """
    print("=" * 60)
    print("HE THONG DU DOAN Y DINH NGUOI DUNG")
    print("   Chatbot Tu Van May Tinh")
    print("=" * 60)
    print()

    # Tai mo hinh
    try:
        model, vectorizer = tai_mo_hinh()
        print("[OK] Da tai mo hinh va vectorizer thanh cong!")
    except FileNotFoundError as e:
        print(e)
        return

    # Cau mau mac dinh
    cau_mau = "Tai chinh 20 trieu nen mua laptop nao?"

    print("\n[INPUT] Du doan cau mau: \"{}\"".format(cau_mau))
    intent = du_doan_intent(cau_mau, model, vectorizer)
    print("[RESULT] Intent du doan : {}".format(intent))
    print("[DESC]   Mo ta          : {}".format(MO_TA_INTENT.get(intent, 'Khong xac dinh')))

    # Vong lap nhap cau moi
    print("\n" + "=" * 60)
    print("Nhap cau hoi de du doan intent (go 'quit' de thoat):")
    print("=" * 60)

    while True:
        try:
            cau_nhap = input("\nBan: ").strip()

            # Kiem tra lenh thoat
            if cau_nhap.lower() in ['quit', 'exit', 'q', 'thoat']:
                print("\nTam biet! Hen gap lai.")
                break

            # Kiem tra cau rong
            if not cau_nhap:
                print("[WARN] Vui long nhap cau hoi!")
                continue

            # Du doan intent
            intent = du_doan_intent(cau_nhap, model, vectorizer)
            print("[RESULT] Intent du doan : {}".format(intent))
            print("[DESC]   Mo ta          : {}".format(MO_TA_INTENT.get(intent, 'Khong xac dinh')))

        except KeyboardInterrupt:
            print("\n\nTam biet!")
            break
        except Exception as e:
            print("[LOI] {}".format(e))


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    main()
