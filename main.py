from vpshunt_detector.download import download_and_unzip
from vpshunt_detector.inference import infer

if __name__ == "__main__":
    weights_dir = download_and_unzip()
    infer(
        weights_dir, "./res/input/4066666_t001_schädel_lat_ventil.png", "./results.csv"
    )
