# Nguồn ảnh siêu âm vú thật

Tổng hợp tháng 11/2026. Trước khi dùng một bộ dữ liệu, hãy đọc giấy phép trên trang gốc. Các bộ dưới đây chỉ được dùng cho nghiên cứu, không dùng để chẩn đoán.

## Đã tích hợp vào dự án

| Bộ | Số ảnh | Mặt nạ | Nguồn | Cách dùng trong dự án |
|---|---|---|---|---|
| **STU-Hospital** (Shantou University, Trung Quốc) | 42 cặp ảnh – mặt nạ | Có | [github.com/xbhlk/STU-Hospital](https://github.com/xbhlk/STU-Hospital) · Zhuang et al., *PLoS ONE* 14(8): e0221535, 2019 | `python manage.py import-stu`; thí nghiệm `real_experiments.py` |
| **BUSI** (Ai Cập) | 682 ảnh sau lọc trùng (780 gốc) | Có | Al-Dhabyani et al., *Data in Brief* 28, 2020 | `python manage.py import-busi`; thí nghiệm R3, R4 |
| **BUS-BRA** (Brazil) | 1.834 ảnh / 1.060 bệnh nhân (1.875 gốc) | Có | Gómez-Flores et al., *Med. Phys.* 51, 2024 · CC BY 4.0 | `python fetch_data.py`; thí nghiệm R4 |
| **BrEaST-Lesions-USG** (Ba Lan) | 251 ảnh (252 có ảnh gốc) | Có | Pawłowska et al., *Sci. Data* 11, 2024 · CC BY 4.0 | `python fetch_data.py`; thí nghiệm R4 |

Repo không chứa sẵn ảnh thật. `import-stu` tải STU từ GitHub; `fetch_data.py` tải BUS-BRA từ Zenodo (hoặc `--mirror`: bản sao không chính thức trên GitHub gồm BUS-BRA và BrEaST, cùng giấy phép CC BY 4.0). Gói ĐATN kèm sẵn tập **kiểm tra** của BUS-BRA và BrEaST (CC BY 4.0, nguồn ghi trong SOURCES.md) để demo offline.

## Có mặt nạ phân đoạn, dùng được cho đề tài

| Bộ | Quy mô | Truy cập | Tham khảo |
|---|---|---|---|
| **BUSI** (Ai Cập) | 780 ảnh (437 lành, 210 ác, 133 bình thường) | Mở, tải trên Mendeley hoặc Kaggle | [Mendeley Data](https://data.mendeley.com/datasets/k8t3gnx9h6/1) · Al-Dhabyani et al., *Data in Brief*, 2020 |
| **BUS-BRA** (Brazil) | 1.875 ảnh / 1.064 bệnh nhân | Mở, tải trên Zenodo | [Zenodo 8231412](https://zenodo.org/records/8231412) · Gómez-Flores et al., *Med. Phys.*, 2024 |
| **BrEaST-Lesions-USG** (Ba Lan) | 256 ảnh, có mặt nạ và mô tả BI-RADS | Mở, tải trên TCIA | [Pawłowska et al., *Sci. Data*, 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10830496/) |
| **UDIAT / Dataset B** (Tây Ban Nha) | 163 ảnh | Phải đăng ký với tác giả | [Yap et al., 2018](https://doi.org/10.1109/JBHI.2017.2731873) · [trang truy cập](https://helward.mmu.ac.uk/STAFF/m.yap/dataset.php) |
| **BUSI_WHU** (Vũ Hán) | 927 ảnh / 816 bệnh nhân | Mở, tải trên Mendeley | [Mendeley v3](https://data.mendeley.com/datasets/k6cpmwybk3/3) |
| **BUS_UC** | 811 cặp ảnh – mặt nạ | Mở, tải trên Mendeley | [Mendeley](https://data.mendeley.com/datasets/3ksd7w7jkx/1) · Iqbal & Sharif, 2024 |
| **BUS-UCLM** (Tây Ban Nha) | 683 cặp ảnh – mặt nạ / 38 bệnh nhân | Mở, tải trên Mendeley | [Mendeley v3](https://data.mendeley.com/datasets/7fvgj4jsp7/3) · Vallez et al., *Sci. Data*, 2025 |
| **QAMEBI** (Iran) | 232 tổn thương | Mở | [qamebi.com](https://qamebi.com/breast-ultrasound-images-database/) |
| **BUSIS** (benchmark) | 562 ảnh | Theo yêu cầu | [Zhang et al., 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9025635/) |

## Việt Nam

| Bộ | Nội dung | Ghi chú |
|---|---|---|
| **HiSBreast** (Bệnh viện Đa khoa tỉnh Cà Mau) | 972 ca siêu âm vú: ảnh, mô tả dấu hiệu tổn thương và kết luận của bác sĩ, bằng tiếng Việt | [Mendeley Data, CC BY 4.0](https://data.mendeley.com/datasets/5c723rpwz2/1). **Không có mặt nạ**, nên dùng được để phân loại, đối chiếu kết luận hoặc thử định tính. Muốn dùng cho phân đoạn thì phải tự vẽ mặt nạ. |
| **VinDr-Mammo** (VinBigData) | 5.000 ca **nhũ ảnh** (X-quang), không phải siêu âm | [Nguyen et al., *Sci. Data*, 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10182079/). Truy cập qua PhysioNet, phải có chứng chỉ. |

Hiện chưa có bộ ảnh siêu âm vú công khai nào của Việt Nam có mặt nạ phân đoạn. Muốn có dữ liệu Việt Nam để phân đoạn, cần hợp tác với bệnh viện: xin phê duyệt đạo đức, ẩn danh ảnh, và nhờ bác sĩ vẽ mặt nạ.

## Danh mục tổng hợp

- [open-medical-ultrasound-datasets](https://github.com/TIanCat/open-medical-ultrasound-datasets): atlas các bộ ảnh siêu âm công khai, được cập nhật liên tục.
- [Ultrasound Open Access Datasets](https://ultrasound-open-access.nidusai.ca/).
- [Awesome-Medical-Dataset – BUSI](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/BUSI.md).
