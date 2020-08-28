# Premise
A combination of various deepfake algoritms to quickly create fake audio and video

# Options
This project has 4 seperate algorithms.
1) [first-order-model](https://github.com/AliaksandrSiarohin/first-order-model): a quick deepfake alorithm that generates a video from a base video and driving image
2) [Speech-Driven Facial Animation](https://github.com/DinoMan/speech-driven-animation): animates a picture to speak an audio input
3) [Real-Time Voice Cloning Toolbox](https://github.com/CorentinJ/Real-Time-Voice-Cloning): a quick text to speech algorithm based off seconds of driving audio
4) [One-shot Voice Conversion](https://github.com/jjery2243542/adaptive_voice_conversion): voice style transform to change the words of one into the words of another

# Setup
## Import
pip install -r requirements.txt

Get from Version Control:
1) https://github.com/AliaksandrSiarohin/first-order-model.git
2) https://github.com/DinoMan/speech-driven-animation.git
3) https://github.com/jjery2243542/adaptive_voice_conversion.git
4) https://github.com/jjery2243542/adaptive_voice_conversion.git

and put these into the local project

## Modify
replace all the dashes in file names with _

Modify each of the files in the following ways:
1) go [here](https://drive.google.com/drive/folders/1PyQJmkdCsAkOYwUyaj_l-l0as-iLDgeH) and download vox-cpk.pth.tar, then place it in first_order_model
2) go [here](https://drive.google.com/drive/folders/1pJdsnknLmMLvA8RQIAV3AQH8vU0FeK16) and download grid.dat, then replace sda/data/grid.dat
3) download: [model](https://drive.google.com/file/d/1n1sPXvT34yXFLT47QZA6FIRGrwMeSsZc/view), delete toolbox/__init__.py
4) download: [model](http://speech.ee.ntu.edu.tw/~jjery2243542/resource/model/is19/vctk_model.ckpt) and [attr](http://speech.ee.ntu.edu.tw/~jjery2243542/resource/model/is19/attr.pkl), add move them into adaptive_voice_conversion

## Alter imports
Next go through each file and correct the imports due to the content root. Go through all the files and add - *project_name.* -  before all necesary imports (if you know a better way, please tell me)
