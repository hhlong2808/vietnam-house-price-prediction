# Dự đoán giá nhà Việt Nam bằng Linear Regression

Dự án cá nhân áp dụng Linear Regression để dự đoán giá nhà tại Việt Nam, 
sử dụng dataset công khai từ Kaggle. Mục tiêu: luyện tập quy trình 
xây dựng một dự án Machine Learning cơ bản từ đầu đến cuối.

## Dataset

[Vietnam Housing Dataset 2024](https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024) 
— ~30,000+ bản ghi bất động sản, gồm các đặc điểm: diện tích, số phòng, 
số tầng, tình trạng pháp lý, địa chỉ, và giá.

## Quy trình

1. **EDA** (`eda.ipynb`): Khám phá phân bố dữ liệu, kiểm tra missing values.
2. **Preprocessing** (`preprocessing.ipynb`):
   - Xóa các cột thiếu quá nhiều dữ liệu (Balcony direction, House direction, 
     Furniture state, Access Road, Frontage).
   - Điền missing values: median cho cột số, gán nhãn hợp lý cho category.
   - Tách City từ Address, chuẩn hóa và gộp nhóm các giá trị ít xuất hiện.
   - Lọc outlier ở Area bằng phương pháp IQR.
   - One-hot encoding cho City và Legal status.
3. **Modeling** (`model.ipynb`): Train/test split (80/20), train Linear Regression, đánh giá.

## Kết quả

| Model | MAE (tỷ) | RMSE (tỷ) | R² |
|---|---|---|---|
| Linear Regression (gốc) | 1.352 | 1.690 | 0.392 |
| + Log-transform Price | 1.376 | 1.759 | 0.341 |

Model gốc (không log-transform) cho kết quả tốt hơn, do phân phối Price 
trong dataset đã tương đối đối xứng sau khi lọc outlier.

![Predicted vs Actual](images/predicted_vs_actual.png)

## Insight chính

- **Vị trí (City) ảnh hưởng đến giá mạnh hơn diện tích (Area)**: hệ số 
  hồi quy của các City lớn hơn nhiều so với Area — cho thấy trong dataset 
  này, vị trí là yếu tố quyết định giá chính, không phải diện tích thô.
- Correlation đơn biến thấp (Area: 0.17) không đồng nghĩa với việc loại 
  bỏ feature — quyết định giữ/bỏ nên dựa vào hệ số thực tế sau khi train 
  model đa biến.

## Cách chạy lại

```bash
pip install -r requirements.txt
```

Dataset gốc đã có sẵn trong folder data 'vietnam_housing_dataset.csv'
`eda.ipynb` → `preprocessing.ipynb` → `model.ipynb`.

## Hướng cải thiện trong tương lai

- Thử các model khác (Random Forest, XGBoost) để so sánh.
- Bổ sung thông tin District chi tiết hơn (thay vì chỉ City).
- Thử thêm feature tương tác (VD: Area × City).