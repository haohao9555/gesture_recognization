# Hand Gesture Recognition with CNN + LSTM

This project classifies hand gesture videos using a simple and beginner-friendly CNN + LSTM pipeline in PyTorch.

## Dataset Structure

Place your dataset like this:

```text
dataset/
    train/
        left/
        right/
        up/
        down/
    val/
        left/
        right/
        up/
        down/
    test/
        left/
        right/
        up/
        down/
```

Each class folder should contain gesture videos such as `.mp4` or `.avi`.
The class label is inferred automatically from the folder name.

## How It Works

Each video is processed as follows:

1. Read the video with OpenCV.
2. Resize frames to `224 x 224`.
3. Uniformly sample exactly `16` frames.
4. If there are fewer than 16 readable frames, repeat the last available frame.
5. Pass each frame through a CNN feature extractor.
6. Feed the sequence of frame features into an LSTM.
7. Use the final LSTM hidden state for classification.

## Tensor Shapes

These are the most important tensor shapes in the model:

1. Dataset output per sample:
   `[16, 3, 224, 224]`

2. Batch from DataLoader:
   `[B, 16, 3, 224, 224]`

3. Before CNN, batch and time are merged:
   `[B * 16, 3, 224, 224]`

4. CNN frame features:
   `[B * 16, 512]`

5. Reshaped back into a sequence:
   `[B, 16, 512]`

6. LSTM final hidden state:
   `[B, hidden_dim]`

7. Classifier output:
   `[B, num_classes]`

## Install

```bash
pip install -r requirements.txt
```

## Prepare Your Own Zip Dataset

If you have a zip file that already contains class folders with videos, you can import it into this project and generate both:

- split videos under `dataset/train|val|test`
- extracted 16-frame samples under `dataset_frames/train|val|test`

Run:

```bash
python prepare_dataset.py --zip-path "D:\path\to\your_dataset.zip" --project-dir . --force
```

This script will:

1. extract the zip into `data/raw_extracted`
2. detect class folders automatically
3. split each class into train/val/test
4. sample exactly 16 frames from every video
5. save frames as `.jpg`

## Train

```bash
python train.py --data-dir dataset --epochs 10 --batch-size 8 --lr 0.001
```

During training, the script prints:

- training loss
- training accuracy
- validation loss
- validation accuracy

The best model is saved based on validation accuracy.

Default checkpoint path:

```text
checkpoints/best_model.pth
```

## Evaluate

To evaluate separately:

```bash
python evaluate.py --data-dir dataset --checkpoint-path checkpoints/best_model.pth --split test
```

This prints:

- loss
- accuracy
- confusion matrix
- classification report

## Files

- `dataset.py`: video loading and 16-frame sampling
- `model.py`: CNN + LSTM model
- `train.py`: training loop and final test evaluation
- `evaluate.py`: standalone evaluation script
- `utils.py`: helper functions

## Notes

- GPU is used automatically if available.
- The model uses `ResNet18` as the frame-level CNN feature extractor.
- For simplicity and offline usability, the code uses `weights=None` by default.
- You can increase epochs or tune the batch size depending on your hardware.
