import whisper
import opencc
import pyaudio
import numpy as np
import queue
import threading
from openvino.runtime import Core

# 加载 OpenVINO 编码器和解码器模型
core = Core()
compiled_encoder_model = core.compile_model(model="weights/base_openvino/base-encoder.xml", device_name="CPU")
compiled_decoder_model = core.compile_model(model="weights/base_openvino/base-decoder.xml", device_name="CPU")

# 获取输入和输出层
encoder_input_layer = compiled_encoder_model.input(0)
print(encoder_input_layer)
encoder_output_layer = compiled_encoder_model.output(0)
print(encoder_output_layer)
decoder_input_layer = compiled_decoder_model.input(0)
print(decoder_input_layer)
decoder_output_layer = compiled_decoder_model.output(0)
print(decoder_output_layer)

# 初始化 opencc 进行繁体到简体转换
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

    # 初始化 pyaudio
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

                    # 准备输入数据
                    mel = whisper.log_mel_spectrogram(audio_data).numpy()
                    mel = np.expand_dims(mel, axis=0)  # 增加批次维度，使形状为 (1, 80, ?)

                    # 使用 OpenVINO 编码器进行推理
                    audio_features = compiled_encoder_model([mel])[encoder_output_layer]
                    
                    # 打印 audio_features 的形状以检查实际形状
                    print(f"audio_features shape: {audio_features.shape}")

                    # 假设 audio_features 的形状为 (6, 1, 253, 512)
                    padded_features = np.zeros((6, 1, 448, 512), dtype=np.float32)
                    padded_features[:, :, :253, :] = audio_features

                    # 准备解码器输入
                    tokens = np.ones((6, 1), dtype=np.int64)

                    # 使用 OpenVINO 解码器进行推理
                    decoder_output = compiled_decoder_model([tokens, padded_features])[decoder_output_layer]

                    # 获取繁体中文文本
                    traditional_text = decoder_output[0]

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
