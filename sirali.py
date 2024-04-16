import subprocess

scripts = {
    "script1": "ray job submit --working-dir /home/ubuntu/demo_deneme/yolov7 -- python train_with_mlflow.py --epochs 1 --device 0 --deterministic --with_mlflow --name HIT_combined-1",
    "script2": "ray job submit --working-dir /home/ubuntu/demo_deneme/yolov7 -- python train_with_mlflow.py --epochs 3 --device 0 --deterministic --with_mlflow --name HIT_combined-3"
    # "script1": "ray job submit --working-dir /home/ray/DEMO/yolov7 -- python train_with_mlflow.py --epochs 1 --device 0 --deterministic --with_mlflow --name HIT-1",
    # "script2": "ray job submit --working-dir /home/ray/DEMO/yolov7 -- python train_with_mlflow.py --epochs 3 --device 0 --deterministic --with_mlflow --name HIT-3"
    # İsterseniz buraya daha fazla script ekleyebilirsiniz
}

# Her bir script'i sırayla çalıştıralım
for script_name, command in scripts.items():
    print(f"{script_name} is starting...")
    process = subprocess.run(command, shell=True)
    #process.wait()  # Bir script işlemi tamamlanana kadar bekleyecektir
    print(f"{script_name} has finished.")


#ray job submit --working-dir /home/ray/ray_deneme/ -- python train_deneme.py 5