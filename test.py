import whisper
import opencc
import pyaudio
import numpy as np
import torch
import queue
import threading

# 加载本地的whisper模型
model_path = "whisper_model/tiny.pt"
model = whisper.load_model(model_path)

# 初始化opencc进行繁体到简体转换
converter = opencc.OpenCC('t2s')

# 实时语音转文本的函数
def real_time_transcribe():
    audio_queue = queue.Queue()
    buffer_size = 1024
    sample_rate = 16000
    format = pyaudio.paInt16
    channels = 1

    def callback(in_data, frame_count, time_info, status):
        audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    # 初始化pyaudio
    p = pyaudio.PyAudio()

    stream = p.open(format=format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=buffer_size,
                    stream_callback=callback)

    stream.start_stream()

    print("开始实时语音转文本...")

    audio_data = np.zeros((0), dtype=np.int16)

    try:
        while True:
            if not audio_queue.empty():
                data = audio_queue.get()
                audio_data = np.append(audio_data, np.frombuffer(data, dtype=np.int16))

                if len(audio_data) > sample_rate * 5:  # 每次处理 5 秒音频
                    # 转换音频为浮点数
                    audio_data = audio_data.astype(np.float32) / 32768.0

                    # 进行语音转文本
                    result = model.transcribe(audio_data)
                    
                    # 获取繁体中文文本
                    traditional_text = result["text"]

                    # 转换文本为简体中文
                    simplified_text = converter.convert(traditional_text)

                    # 打印简体中文文本
                    print(simplified_text)

                    # 清空音频缓冲区
                    audio_data = np.zeros((0), dtype=np.int16)
    except KeyboardInterrupt:
        print("结束实时语音转文本")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# 开始实时语音转文本
real_time_transcribe()


