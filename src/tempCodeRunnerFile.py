model_path = "C:\Python_Projects\\fintwin\\runs\\train7\weights\\best.pt"
image_path = "C:\python projects\\YOLO\\FinTwin360\\ft_dataset\\images\\train\\IMG_20240722_205813.jpeg"
image_path_2 = "C:\python projects\\YOLO\\FinTwin360\\ft_dataset\images\\train\IMG_20240619_174837.JPEG"

data_extract = ocr(model_path)
datas = data_extract.get_receipt_info(image_path_2)

print(datas[0])