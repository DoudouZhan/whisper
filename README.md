# Whisper
This repo is forked from [OPENAI WHISPER](https://github.com/openai/whisper).


## Get started
- step 1. clone the repo
    ```sh
    git clone https://github.com/jianboma/whisper
    cd whisper
    ```
- step 2. create virtual environment and install 
    ```sh
    # you can use conda to create the virtual envrionment
    conda create -n whisper python=3.10
    # install editable model with whisper repo
    pip install -e .
    
    ```
- step 3. try [test_whisper.py](test_sample/test_whisper.py) with the installed environment


## test with librispeech-test-clean

To verify the librispeech, there are several steps.
- step 1. install dependencies
    ```sh
    pip install -r test_librispeech_requirements.txt
    ```
- step 2. download [librispeech](https://www.openslr.org/12). You can download one subset, e.g. test-clean. 
- step 3. run [test_whisper_librispeech.py](test_sample/test_whisper_librispeech.py)
    ```sh
    python test_sample/test_whisper_librispeech.py --librispeech-root=your_librispeech_root --librispeech-subset=test-clean --whisper-model=base.en --whisper-weights-root=your_whisper_weights_folder
    ```
    `your_librispeech_root` is the one before `LibriSpeech`.
    **NOTE: the weights of the corresponding whisper model will automatically download.**

- step 4. the output should be,
    ```sh
    WER: 4.27 %
    ```