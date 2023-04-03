import os
import json

import torch
from PIL import Image
from PIL.ImageQt import ImageQt
from torchvision import transforms


from model import convnext_tiny as create_model

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class AuroraRecognition(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Aurora Recognition')
        self.setGeometry(300, 300, 400, 400)
        layout = QVBoxLayout()

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.browse_button = QPushButton("Browse Image")
        self.browse_button.clicked.connect(self.browse_image)
        layout.addWidget(self.browse_button)

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict_image)
        layout.addWidget(self.predict_button)

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"using {self.device} device.")

        self.num_classes = 6
        self.img_size = 224
        self.data_transform = transforms.Compose(
            [transforms.Resize(int(self.img_size * 1.14)),
             transforms.CenterCrop(self.img_size),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.json_path = './class_indices.json'
        assert os.path.exists(self.json_path), "file: '{}' dose not exist.".format(self.json_path)

        with open(self.json_path, "r") as f:
            self.class_indict = json.load(f)

        self.model = create_model(num_classes=self.num_classes).to(self.device)
        self.model_weight_path = "./weights/best_model.pth"
        self.model.load_state_dict(torch.load(self.model_weight_path, map_location=self.device))
        self.model.eval()

    def browse_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if filename:
            self.image = Image.open(filename).convert('RGB')
            self.image_label.setPixmap(QPixmap.fromImage(ImageQt(self.image)))
            self.image_label.setScaledContents(True)

    def predict_image(self):
        try:
            img = self.data_transform(self.image)
            img = torch.unsqueeze(img, dim=0)

            with torch.no_grad():
                output = torch.squeeze(self.model(img.to(self.device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            print_res = "class: {}   prob: {:.3}".format(self.class_indict[str(predict_cla)],
                                                         predict[predict_cla].numpy())
            self.result_label.setText(print_res)
        except AttributeError:
            QMessageBox.warning(self, "Warning", "Please select an image first.")
            return


if __name__ == '__main__':
    app = QApplication([])
    aurora_recognition = AuroraRecognition()
    aurora_recognition.show()
    app.exec_()
