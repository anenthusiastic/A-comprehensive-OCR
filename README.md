# Comprehensive OCR Application (Kapsamlı OCR Uygulaması)

Bu proje, Python ve Tesseract OCR motorunu kullanarak geliştirilmiş, kapsamlı bir Optik Karakter Tanıma (OCR) ve görüntü işleme aracıdır. Sadece düz metin okumakla kalmaz, aynı zamanda kredi kartı numarası ayıklama, plaka tanıma ve metin içi arama gibi özelleşmiş fonksiyonlar sunar.

## 🎯 Özellikler

Uygulama, farklı kullanım senaryolarına yönelik modüler çözümler sunar:
* **Grafiksel Kullanıcı Arayüzü (GUI):** Komut satırı ile uğraşmadan, görsel arayüz üzerinden dosya seçimi ve işlem yapma imkanı.
* **Metin Çıkarma (Text Extraction):** Görüntülerdeki metinleri yüksek doğrulukla dijital metne çevirir.
* **Belge Dönüştürme:** Okunan metinleri `.doc` (Word) formatında dışa aktarabilir.
* **Kredi Kartı Okuma:** Görüntü üzerindeki kredi kartı numaralarını tespit eder ve filtreler.
* **Plaka Tanıma:** Araç görsellerinden plaka tespiti ve metin dönüşümü yapar.
* **Metin Arama:** Görüntü içerisinde belirli bir kelimeyi veya metin öbeğini arayabilir.
* **Görüntü İşleme:** OCR doğruluğunu artırmak için gürültü azaltma (noise reduction) ve eşikleme (thresholding) gibi ön işleme teknikleri uygular.

## 🛠 Kullanılan Teknolojiler

* **Tkinter:** Kullanıcı arayüzü için.
* **Python 3.x:** Ana programlama dili.
* **OpenCV (`cv2`):** Görüntü ön işleme (preprocessing) işlemleri için.
* **Tesseract OCR (`pytesseract`):** Metin tanıma motoru.
* **PIL (Pillow):** Görüntü manipülasyonu için.
* **Matplotlib:** Sonuçların görselleştirilmesi için.

## 🚀 Kurulum ve Kullanım

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyin:
    
### 1. Gereksinimler

 **Depoyu Klonlayın:**
    ```bash
    git clone [https://github.com/anenthusiastic/PyTextVision.git](https://github.com/anenthusiastic/PyTextVision.git)
    cd PyTextVision
    ```
  
**Gereksinimleri Yükleyin:**
    ```bash
    pip install opencv-python pytesseract Pillow
    ```

**Tesseract Kurulumu:**
    Sisteminizde [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) yüklü olmalıdır.

### 2. Çalıştırma
  Proje dizinine gidin ve ana uygulamayı başlatın:
  
  ```bash
  python app.py
  ```

### 3. 💻 Kullanım
  Açılan pencereden bir görsel seçin ve "Metni Çıkar" (Extract Text) butonuna tıklayın.

## 📸 Ekran Görüntüleri

![jyhgjk](https://user-images.githubusercontent.com/67736718/125866214-54ea57f5-5f8b-4bfc-a068-7f54e3ed26ac.png)
Extracting credit card number from credit card image


![ıyjuthygt](https://user-images.githubusercontent.com/67736718/125866327-7e603dcf-579f-473d-82e7-a073612d3600.png)
Vehicle plate recognition


![tetx](https://user-images.githubusercontent.com/67736718/125866414-95940d03-e212-4b17-b5a3-a975611c8aa3.png)
String searching on the text-only image


![jhtetk](https://user-images.githubusercontent.com/67736718/125866449-57a4ba83-0c37-4961-8a92-3a7104e5e2a1.png)
Extracting text from image to .doc file
