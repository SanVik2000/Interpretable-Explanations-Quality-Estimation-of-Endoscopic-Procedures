# Interpretable Explanations and Quality Estimation of Medical Endoscopic Procedures

This work is part of my Dual-Degree project under the guidance of [Prof. Chandrasekhar lakshminarayan](https://sites.google.com/view/chandrashekar-lakshminarayanan) and [Prof. Arun Rajkumar](https://sites.google.com/view/arun-rajkumar).
This work aims at explaining black-box abstraction of video-based deep learning models which provide insights into faulty procedures.

## Problem Setting

Given a binary-classification dataset containing videos of endoscopic procedures, identify the temporal and spartial locations which are responsible for the model's ability to identify errors.
Specifically, the videos represent intubation procedures, and the excerpt is considererd to be faulty, if intubation progresses into the trachea.

## Experiments

Execute the following code to train a video-classification model using different network architectures.

```bash
bash run.sh
```

For explaining a trained model (say CNN-Transformer), execute the following command.

```bash
python3 Main.py --root_dir /media/sanvik/Data/Dual_Degree_Project/ --model_net CNNTrans --out_dir CNNTransPool_4_128_CLS --head_type cls --test --vis
```
## Results

Sample1 <br>
![Visual Explanations generated using CNN_Transformer](/assets/explanation1.jpg "Sample 1") <br><br><br>
Sample2 <br>
![Visual Explanations generated using CNN_Transformer](/assets/explanation2.jpg "Sample 2")

- The first row indicates the raw-attention scores - Brighter colors (yellow) indicate higher pairwise relation with the final prediction
- Second row indicates the raw video frames
- Third row represents the attention scores overlayed over video frames
- Fourth row shows the GradCam visualizations over each frame
