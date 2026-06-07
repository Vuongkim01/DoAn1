"""
Module tien xu ly van ban tieng Viet.

Module nay cung cap cac ham de tien xu ly van ban tieng Viet
phuc vu cho bai toan phan loai y dinh (intent classification).
Cac buoc tien xu ly bao gom:
    - Chuyen chu thuong
    - Loai bo ky tu dac biet
    - Tach tu bang underthesea
    - Loai bo stopwords co ban tieng Viet
"""

import re
import sys

# Fix loi encoding tren Windows (cp1252 khong ho tro tieng Viet)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from underthesea import word_tokenize

# Danh sach stopwords co ban tieng Viet
# Day la cac tu xuat hien thuong xuyen nhung khong mang nhieu y nghia
VIETNAMESE_STOPWORDS = {
    "va", "cua", "la", "co", "duoc", "cho", "voi", "cac", "mot", "nhung",
    "trong", "da", "nay", "do", "tu", "khi", "tai", "de", "theo", "ve",
    "lai", "tren", "vao", "ra", "con", "den", "nhu", "thi", "cung",
    "se", "rat", "hay", "hoac", "neu", "nhung", "bi", "do", "qua",
    "vi", "nen", "ma", "dang", "o", "the", "ai", "gi", "nao", "dau",
    "sao", "thi", "a", "nhé", "nhi", "nha", "vay", "ay",
    "roi", "moi", "chi", "deu", "phai", "hon", "len", "xuong", "sang",
    # Tieng Viet co dau
    "và", "của", "là", "có", "được", "cho", "với", "các", "một", "những",
    "trong", "đã", "này", "đó", "từ", "khi", "tại", "để", "theo", "về",
    "lại", "trên", "vào", "ra", "còn", "đến", "như", "thì", "cũng",
    "sẽ", "rất", "hay", "hoặc", "nếu", "nhưng", "bị", "do", "qua",
    "vì", "nên", "mà", "đang", "ở", "thế", "ai", "gì", "nào", "đâu",
    "sao", "thì", "à", "ạ", "ơi", "nhé", "nhỉ", "nha", "vậy", "ấy",
    "rồi", "mới", "chỉ", "đều", "phải", "hơn", "lên", "xuống", "sang",
    "bạn", "tôi", "mình", "chúng", "họ", "em", "anh", "chị", "bạn_ơi"
}


def chuyen_chu_thuong(text):
    """
    Chuyen toan bo van ban sang chu thuong.

    Args:
        text (str): Van ban dau vao.

    Returns:
        str: Van ban da duoc chuyen sang chu thuong.

    Vi du:
        >>> chuyen_chu_thuong("Laptop ASUS Rat TOT")
        'laptop asus rat tot'
    """
    return text.lower()


def loai_bo_ky_tu_dac_biet(text):
    """
    Loai bo cac ky tu dac biet, chi giu lai chu cai, so va khoang trang.

    Su dung regex de loai bo tat ca ky tu khong phai chu cai tieng Viet,
    so hoac khoang trang.

    Args:
        text (str): Van ban dau vao.

    Returns:
        str: Van ban da loai bo ky tu dac biet.

    Vi du:
        >>> loai_bo_ky_tu_dac_biet("Laptop gia 20.000.000d!!!")
        'Laptop gia 20000000d'
    """
    # Giu lai chu cai (bao gom tieng Viet co dau), so va khoang trang
    text = re.sub(r'[^\w\sàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]',
                  ' ', text, flags=re.IGNORECASE)
    # Loai bo khoang trang thua
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tach_tu(text):
    """
    Tach tu tieng Viet su dung thu vien underthesea.

    Underthesea su dung mo hinh NLP de tach tu ghep tieng Viet
    (vi du: "may tinh" -> "may_tinh").

    Args:
        text (str): Van ban dau vao.

    Returns:
        list: Danh sach cac tu da duoc tach.

    Vi du:
        >>> tach_tu("toi muon mua may tinh xach tay")
        ['toi', 'muon', 'mua', 'may_tinh', 'xach_tay']
    """
    return word_tokenize(text, format="text").split()


def loai_bo_stopwords(words):
    """
    Loai bo cac stopwords tieng Viet khoi danh sach tu.

    Stopwords la cac tu xuat hien thuong xuyen nhung khong mang
    nhieu y nghia cho viec phan loai (vi du: "va", "cua", "la").

    Args:
        words (list): Danh sach cac tu can loc.

    Returns:
        list: Danh sach tu da loai bo stopwords.

    Vi du:
        >>> loai_bo_stopwords(["toi", "muon", "mua", "laptop"])
        ['muon', 'mua', 'laptop']
    """
    return [word for word in words if word not in VIETNAMESE_STOPWORDS]


def tien_xu_ly(text):
    """
    Ham tien xu ly chinh - thuc hien toan bo pipeline tien xu ly.

    Pipeline xu ly theo thu tu:
        1. Chuyen chu thuong
        2. Loai bo ky tu dac biet
        3. Tach tu bang underthesea
        4. Loai bo stopwords
        5. Ghep lai thanh chuoi

    Args:
        text (str): Van ban tieng Viet dau vao.

    Returns:
        str: Chuoi da duoc tien xu ly, cac tu cach nhau boi khoang trang.

    Vi du:
        >>> tien_xu_ly("Toi muon mua laptop de hoc lap trinh, ban tu van giup!")
        'muon mua laptop hoc lap_trinh tu_van giup'
    """
    # Buoc 1: Chuyen chu thuong
    text = chuyen_chu_thuong(text)

    # Buoc 2: Loai bo ky tu dac biet
    text = loai_bo_ky_tu_dac_biet(text)

    # Buoc 3: Tach tu tieng Viet
    words = tach_tu(text)

    # Buoc 4: Loai bo stopwords
    words = loai_bo_stopwords(words)

    # Buoc 5: Ghep lai thanh chuoi
    return " ".join(words)


# ==========================================
# Chay thu module neu chay truc tiep
# ==========================================
if __name__ == "__main__":
    # Cac cau mau de kiem tra
    cau_mau = [
        "Toi muon mua laptop de hoc lap trinh, ban tu van giup!",
        "Laptop Asus ROG gia bao nhieu tien???",
        "MacBook Pro M3 co bao nhieu RAM va dung luong SSD?",
        "So sanh Dell XPS va MacBook Air cai nao tot hon?"
    ]

    print("=" * 60)
    print("KIEM TRA MODULE TIEN XU LY TIENG VIET")
    print("=" * 60)

    for cau in cau_mau:
        print(f"\n[GOC]   : {cau}")
        print(f"[XU LY] : {tien_xu_ly(cau)}")
        print("-" * 60)
