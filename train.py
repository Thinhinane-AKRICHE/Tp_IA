import sys
import os
import random
from roboflow import Roboflow
from ultralytics import YOLO
import yaml
import time


class Main:
    rf: Roboflow
    project: object
    dataset: object
    model: object
    results: object
    model_size: str

    def __init__(self):
        self.model_size = sys.argv[6]
        self.import_dataset()
        self.train()

    def import_dataset(self):
        self.rf = Roboflow(api_key=sys.argv[1])
        self.project = self.rf.workspace(sys.argv[2]).project(sys.argv[3])
        self.dataset = self.project.version(sys.argv[4]).download("yolov8-obb")

        with open(f"{self.dataset.location}/data.yaml", "r") as file:
            data = yaml.safe_load(file)

        data["path"] = self.dataset.location

        with open(f"{self.dataset.location}/data.yaml", "w") as file:
            yaml.dump(data, file, sort_keys=False)

    def train(self):
        list_of_models = ["n", "s", "m", "l", "x"]
        if self.model_size != "ALL" and self.model_size in list_of_models:
            self.model = YOLO(f"yolov8{self.model_size}-obb.pt")

            self.results = self.model.train(
                data=f"{self.dataset.location}/" f"yolov8-obb.yaml",
                epochs=int(sys.argv[5]),
                imgsz=640,
            )

            self.test()

        elif self.model_size == "ALL":
            for model_size in list_of_models:
                self.model = YOLO(f"yolov8{model_size}.pt")

                self.results = self.model.train(
                    data=f"{self.dataset.location}" f"/yolov8-obb.yaml",
                    epochs=int(sys.argv[5]),
                    imgsz=640,
                )

                self.test()

        else:
            print("Invalid model size")

    def test(self):
        print("Testing the model in 10 seconds")
        time.sleep(10)
        name_of_last_folder = os.listdir("runs/detect")[-1]
        model = YOLO(f"runs/detect/{name_of_last_folder}/weights/best.pt")

        random_file = random.choice(os.listdir(f"{self.dataset.location}/test/images"))
        file_name = os.path.join(f"{self.dataset.location}/test/images", random_file)

        results = model(file_name)

        for result in results:
            result.show()


if __name__ == "__main__":
    Main()