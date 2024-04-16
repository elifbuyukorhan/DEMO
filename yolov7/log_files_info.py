import os
import yaml
import cv2
import logging
import mlflow
import tensorflow as tf
from mlflow import log_metric, log_params


LOGGER = logging.getLogger(__name__)

def colorstr(*input):
    """
    This function colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, 
    i.e.  colorstr('blue', 'hello world')
    """

    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def log_folder_info(root_folder):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            LOGGER.info(f"Logging info for folder: {folder_path}")

            events_files = [f for f in os.listdir(folder_path) if f.startswith('events.out.tfevents')]

            if not events_files:
                LOGGER.warning(f"No events file found in folder: {folder_path}")
                continue

            events_file = os.path.join(folder_path, events_files[0])  # Use the first found events file
            hyp_file = os.path.join(folder_path, 'hyp.yaml')
            opt_file = os.path.join(folder_path, 'opt.yaml')

            if not (os.path.exists(hyp_file) and os.path.exists(opt_file)):
                LOGGER.warning(f"Missing necessary files in folder: {folder_path}")
                continue

            with open(hyp_file, 'r') as file:
                hyp = yaml.safe_load(file)

            with open(opt_file, 'r') as file:
                opt = yaml.safe_load(file)

            info_dict = {
                'input size': opt['img_size'],
                'batch_size': opt['batch_size'],
                'epoch': opt['epochs'],
                'rect': opt['rect'],
                'freeze': opt['freeze'],
                'adam': opt['adam'],
                'weights': opt['weights']
            }

            event_iterator = tf.compat.v1.train.summary_iterator(events_file)

            mlflow_location = 'http://127.0.0.1:5005'
            mlflow.set_tracking_uri(mlflow_location)

            _expr_name = 'yolov7-tiny'
            experiment = mlflow.get_experiment_by_name(_expr_name)

            run_name = folder_name  # Using folder name as run name

            if experiment is None:
                mlflow.create_experiment(_expr_name)
            mlflow.set_experiment(_expr_name)

            prefix = colorstr('MLFlow: ')
            try:
                _mlflow, mlflow_active_run = mlflow, None if not mlflow else mlflow.start_run(run_name=run_name)
                if mlflow_active_run is not None:
                    _run_id = mlflow_active_run.info.run_id
                    LOGGER.info(f'{prefix}Using run_id({_run_id}) at {mlflow_location}')
            except Exception as err:
                LOGGER.error(f'{prefix}Failing init - {repr(err)}')
                LOGGER.warning(f'{prefix}Continuing without mlflow')
                _mlflow = None
                mlflow_active_run = None

            step = 0

            for event in event_iterator:
                if len(event.summary.value) > 0:
                    for value in event.summary.value:
                        tag = value.tag.replace("/", "_").replace(":", "_")
                        if value.HasField('simple_value') and not tag.startswith("x_lr"):
                            step = event.step
                            log_metric(tag, value.simple_value, step=step)

                    if hasattr(event, 'wall_time'):
                        log_metric('timewall', event.wall_time, step=step)

            mlflow.log_params(info_dict)
            mlflow.log_params(hyp)

            conf_path = os.path.join(folder_path, 'confusion_matrix.png')
            pr_curve_path = os.path.join(folder_path, 'PR_curve.png')
            f1_curve_path = os.path.join(folder_path, 'F1_curve.png')

            conf = cv2.imread(conf_path)
            pr_curve = cv2.imread(pr_curve_path)
            f1_curve = cv2.imread(f1_curve_path)

            mlflow.log_image(conf, 'confusion_matrix.png')
            mlflow.log_image(pr_curve, 'PR_curve.png')
            mlflow.log_image(f1_curve, 'F1_curve.png')

            best_path = os.path.join(folder_path, 'best.pt')
            mlflow.log_artifact(best_path)
            model_uri = f'runs:/{_run_id}/'
            mlflow.register_model(model_uri, run_name)

            mlflow.end_run()

            LOGGER.info(f"Logging completed for folder: {folder_path}")

    LOGGER.info("All folders processed.")


# Kullanımı
root_folder = '/home/vision/Elif/PIPELINE/YOLOv7-tiny'
log_folder_info(root_folder)
print("Metrics successfully logged to MLflow for all folders.")
