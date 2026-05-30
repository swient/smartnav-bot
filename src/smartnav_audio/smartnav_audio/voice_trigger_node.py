#!/usr/bin/env python3
"""語音喚醒觸發節點

使用 VAD 和 KWS 實現語音喚醒
監聽麥克風，在檢測到喚醒詞時發佈音訊
"""

import threading
import sherpa_onnx
import unicodedata
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Dict, Optional
from pypinyin import pinyin, Style

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import Bool

from smartnav_msgs.msg import AudioData
from smartnav_audio.voice_utils import AudioCodec, get_model_path, validate_audio_data
from smartnav_audio.audio_recorder import AudioRecorder


class TriggerState(Enum):
    """三階段狀態列舉"""

    IDLE = "idle"  # 第一階段：休眠監測
    SPOTTING = "spotting"  # 第二階段：喚醒詞驗證
    COMMAND = "command"  # 第三階段：指令解析


class VoiceTriggerNode(Node):
    """語音喚醒觸發節點

    使用 VAD 和 KWS 實現語音喚醒功能

    Publishers:
        /audio_in (smartnav_msgs/AudioData): 音訊數據流
        /voice_triggered (std_msgs/Bool): 喚醒觸發事件
    """

    def __init__(self) -> None:
        """初始化語音喚醒觸發節點

        宣告所有必要參數，初始化音訊處理引擎
        """
        super().__init__("voice_trigger_node")

        # ============ 參數聲明 ============

        # 基本音訊參數
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("chunk_size", 512)
        self.declare_parameter("device", -1)
        self.declare_parameter("dtype", "float32")
        self.declare_parameter("output_format", "pcm_s16le")

        # VAD 參數
        self.declare_parameter("vad_num_threads", 2)

        # KWS 參數
        self.declare_parameter("kws_keyword", "小派")
        self.declare_parameter("kws_num_threads", 2)

        # 狀態轉移參數 (毫秒)
        self.declare_parameter("speech_start_timeout", 100)
        self.declare_parameter("silence_timeout", 500)
        self.declare_parameter("command_initial_wait_timeout", 2000)

        # Topic 名稱參數
        self.declare_parameter("audio_topic", "/audio_in", ParameterDescriptor(description="音訊數據流話題"))
        self.declare_parameter("triggered_topic", "/voice_triggered", ParameterDescriptor(description="喚醒觸發話題"))

        # ============ 參數獲取 ============

        # 音訊參數
        self.sample_rate: int = self.get_parameter("sample_rate").get_parameter_value().integer_value
        self.chunk_size: int = self.get_parameter("chunk_size").get_parameter_value().integer_value
        self.device: int = self.get_parameter("device").get_parameter_value().integer_value
        self.dtype: str = self.get_parameter("dtype").get_parameter_value().string_value
        self.output_format: str = self.get_parameter("output_format").get_parameter_value().string_value

        # VAD 參數
        self.vad_num_threads: int = self.get_parameter("vad_num_threads").get_parameter_value().integer_value

        # KWS 參數
        self.kws_keyword_raw: str = self.get_parameter("kws_keyword").get_parameter_value().string_value
        self.kws_num_threads: int = self.get_parameter("kws_num_threads").get_parameter_value().integer_value

        # 狀態轉移參數 (毫秒轉幀數)
        speech_start_timeout_ms: int = self.get_parameter("speech_start_timeout").get_parameter_value().integer_value
        silence_timeout_ms: int = self.get_parameter("silence_timeout").get_parameter_value().integer_value
        command_initial_wait_timeout_ms: int = (
            self.get_parameter("command_initial_wait_timeout").get_parameter_value().integer_value
        )

        # Topic 名稱參數
        audio_topic: str = self.get_parameter("audio_topic").get_parameter_value().string_value
        triggered_topic: str = self.get_parameter("triggered_topic").get_parameter_value().string_value

        # 轉換毫秒為幀數 (frames = ms * sample_rate / 1000 / self.chunk_size)
        self.speech_start_frames: int = max(1, int(speech_start_timeout_ms * self.sample_rate / 1000 / self.chunk_size))
        self.silence_frames_threshold: int = max(1, int(silence_timeout_ms * self.sample_rate / 1000 / self.chunk_size))
        self.command_initial_wait_frames: int = max(
            1, int(command_initial_wait_timeout_ms * self.sample_rate / 1000 / self.chunk_size)
        )

        # sherpa-onnx 初始化
        self.kws_keyword: str = ""
        self.vad: Optional[sherpa_onnx.VadModel] = None
        self.kws: Optional[sherpa_onnx.OnlineRecognizer] = None
        self._init_sherpa_onnx()

        # 三階段狀態機
        self._state = TriggerState.IDLE
        self._lock = threading.Lock()

        # Idle 階段：連續 VAD 正幀計數，用於檢測語音起點
        self._vad_positive_frames: int = 0

        # Spotting 階段：累積音訊流和靜音幀計數
        self._kws_stream: Optional[sherpa_onnx.OnlineStream] = None
        self._silence_frames: int = 0

        # Command 階段：初始等待計數器和說話結束檢測
        self._command_wait_frames: int = 0
        self._in_command_initial_wait: bool = False

        # 麥克風錄製器初始化
        self._recorder: Optional[AudioRecorder] = None
        self._init_recorder()

        # 建立 QoS 配置檔
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.audio_pub = self.create_publisher(AudioData, audio_topic, best_effort_qos)
        self.triggered_pub = self.create_publisher(Bool, triggered_topic, reliable_qos)

        self.get_logger().info("✓ 語音喚醒觸發節點已初始化")
        self.get_logger().info(f"  發布話題: {audio_topic}")
        self.get_logger().info(f"  發布話題: {triggered_topic}")
        self.get_logger().info(f"  採樣率: {self.sample_rate} Hz")
        self.get_logger().info(f"  語音起點檢測: {speech_start_timeout_ms} ms ({self.speech_start_frames} 幀)")
        self.get_logger().info(f"  靜音結束檢測: {silence_timeout_ms} ms ({self.silence_frames_threshold} 幀)")
        self.get_logger().info(
            f"  Command 初始等待: {command_initial_wait_timeout_ms} ms ({self.command_initial_wait_frames} 幀)"
        )

    def _init_sherpa_onnx(self) -> None:
        """初始化 sherpa-onnx 引擎

        載入 VAD 和 KWS 模型，並轉換喚醒關鍵字為音素格式
        """
        try:
            # 初始化 VAD
            try:
                vad_model_dir = get_model_path("vad")
                if vad_model_dir:
                    vad_model_path = vad_model_dir / "silero_vad.onnx"
                    if vad_model_path.exists():
                        vad_config = sherpa_onnx.VadModelConfig()
                        vad_config.silero_vad.model = str(vad_model_path)
                        vad_config.sample_rate = self.sample_rate
                        vad_config.num_threads = self.vad_num_threads

                        self.vad = sherpa_onnx.VadModel.create(vad_config)
                        self.get_logger().info("✓ VAD 模型已載入")
                    else:
                        self.get_logger().warning(f"✗ VAD 模型文件不存在: {vad_model_path}")
                else:
                    self.get_logger().warning("✗ 未找到 VAD 模型目錄")
            except Exception as e:
                self.get_logger().warning(f"✗ 載入 VAD 模型失敗: {e}")

            # 初始化 KWS
            try:
                kws_model_dir = get_model_path("kws")
                if kws_model_dir:
                    encoder_file = kws_model_dir / "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"
                    decoder_file = kws_model_dir / "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"
                    joiner_file = kws_model_dir / "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"
                    tokens_file = kws_model_dir / "tokens.txt"

                    if all(f.exists() for f in [encoder_file, decoder_file, joiner_file, tokens_file]):
                        self.kws = sherpa_onnx.OnlineRecognizer.from_transducer(
                            tokens=str(tokens_file),
                            encoder=str(encoder_file),
                            decoder=str(decoder_file),
                            joiner=str(joiner_file),
                            num_threads=self.kws_num_threads,
                            sample_rate=self.sample_rate,
                            decoding_method="greedy_search",
                            enable_endpoint_detection=False,
                        )
                        self.get_logger().info(f"✓ KWS 模型已載入")

                        # 載入英文音素表
                        phone_dict = self._load_english_phone_dict(kws_model_dir)

                        # 轉換關鍵字為音素格式
                        self.kws_keyword = self._convert_keyword_to_phonemes(self.kws_keyword_raw, phone_dict)
                        if self.kws_keyword:
                            self.get_logger().info(
                                f"✓ 喚醒關鍵字轉換: '{self.kws_keyword_raw}' -> '{self.kws_keyword}'"
                            )
                        else:
                            self.get_logger().error(f"✗ 喚醒關鍵字轉換失敗")
                    else:
                        self.get_logger().warning(f"✗ KWS 模型文件不完整: {kws_model_dir}")
                else:
                    self.get_logger().warning("✗ 未找到 KWS 模型目錄")
            except Exception as e:
                self.get_logger().warning(f"✗ 載入 KWS 模型失敗: {e}")
        except Exception as e:
            self.get_logger().error(f"✗ 初始化 sherpa-onnx 失敗: {e}")

    def _init_recorder(self) -> None:
        """初始化並啟動麥克風錄製器

        建立 AudioRecorder 實例並啟動音訊捕獲
        """
        try:
            self._recorder = AudioRecorder(
                sample_rate=self.sample_rate,
                chunk_size=self.chunk_size,
                device=self.device,
                audio_callback=self.process_audio_chunk,
                logger=self.get_logger(),
            )
            self._recorder.start()
            self.get_logger().info("✓ 麥克風錄製已啟動")
        except Exception as e:
            self.get_logger().error(f"✗ 初始化麥克風錄製器失敗: {e}")

    def _stop_recorder(self) -> None:
        """停止麥克風錄製

        停止音訊捕獲並釋放資源
        """
        if self._recorder:
            try:
                self._recorder.stop()
                self.get_logger().info("✓ 麥克風錄製已停止")
            except Exception as e:
                self.get_logger().error(f"✗ 停止麥克風錄製失敗: {e}")

    def _set_state(self, new_state: TriggerState) -> None:
        """轉移到新狀態

        Args:
            new_state: 目標狀態
        """
        with self._lock:
            if self._state == new_state:
                return

            old_state = self._state
            self._state = new_state

            # 日誌輸出
            transition_str = f"{old_state.value.upper()} -> {new_state.value.upper()}"
            self.get_logger().info(f"[狀態轉移] {transition_str}")

            # 狀態開始初始化
            self._on_state_enter(new_state)

    def _on_state_enter(self, state: TriggerState) -> None:
        """處理進入新狀態時的初始化

        根據新狀態重置相關計數器和狀態變數

        Args:
            state: 新狀態
        """
        if state == TriggerState.IDLE:
            self._vad_positive_frames = 0
            self._kws_stream = None
            self._silence_frames = 0
            self._command_wait_frames = 0
            self._in_command_initial_wait = False
        elif state == TriggerState.SPOTTING:
            self._kws_stream = None
            self._silence_frames = 0
        elif state == TriggerState.COMMAND:
            self._kws_stream = None
            self._silence_frames = 0
            self._command_wait_frames = 0
            self._in_command_initial_wait = True

    def _publish_audio(self, audio: np.ndarray, is_final: bool = False) -> None:
        """發佈音訊數據

        Args:
            audio: 音訊數據 (float32 numpy array)
            is_final: 是否為最終語音塊（語音已結束）
        """
        try:
            valid, error = validate_audio_data(audio, self.sample_rate)
            if not valid:
                self.get_logger().warning(f"✗ 無效的音訊數據，無法發佈: {error}")
                return

            if self.output_format == "pcm_s16le":
                audio_bytes = AudioCodec.encode_pcm_s16le(audio)
                format_str = "pcm_s16le"
            elif self.output_format == "pcm_s32le":
                audio_bytes = AudioCodec.encode_pcm_s32le(audio)
                format_str = "pcm_s32le"
            elif self.output_format == "pcm_f32le":
                audio_bytes = AudioCodec.encode_pcm_f32le(audio)
                format_str = "pcm_f32le"
            else:
                self.get_logger().warning(f"✗ 不支援的音訊格式: {self.output_format}")
                return

            msg = AudioData()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "microphone"
            msg.data = audio_bytes
            msg.format = format_str
            msg.sample_rate = self.sample_rate
            msg.channels = 1
            msg.is_final = is_final

            self.audio_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"✗ 發佈音訊失敗: {e}")

    def process_audio_chunk(self, audio: np.ndarray) -> bool:
        """處理音訊塊並進行三階段狀態機轉移

        執行喚醒檢測，根據當前狀態進行相應的處理，發佈音訊或觸發事件

        Args:
            audio: 音訊數據 (float32 numpy array)

        Returns:
            bool: 是否觸發喚醒
        """
        triggered = False

        # 確保 audio 為 float32
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            else:
                audio = audio.astype(np.float32)

        # 執行 VAD
        speech_detected = False
        if self.vad:
            try:
                audio_list = audio.tolist()
                speech_detected = self.vad.is_speech(audio_list)

            except Exception as e:
                self.get_logger().error(f"✗ VAD 檢測失敗: {e}")
                return False

        # ============ 三階段狀態機邏輯 ============

        # Idle 階段
        with self._lock:
            current_state = self._state

        if current_state == TriggerState.IDLE:
            if speech_detected:
                self._vad_positive_frames += 1
                self.get_logger().debug(f"[IDLE] VAD 正幀計數: {self._vad_positive_frames}/{self.speech_start_frames}")
                if self._vad_positive_frames >= self.speech_start_frames:
                    # VAD 連續正幀達到閾值，轉入 Spotting 階段
                    self._set_state(TriggerState.SPOTTING)
                    self.get_logger().info(f"語音起點檢測 (連續 {self._vad_positive_frames} 幀語音)")
            else:
                if self._vad_positive_frames > 0:
                    self.get_logger().debug(f"[IDLE] VAD 正幀計數重置 (從 {self._vad_positive_frames} 幀)")
                self._vad_positive_frames = 0

        # Spotting 階段
        with self._lock:
            current_state = self._state

        if current_state == TriggerState.SPOTTING:
            if speech_detected:
                self._silence_frames = 0
                # 在 VAD 偵測的語音期間，持續累積音訊到 KWS 流
                self._accumulate_kws_audio(audio)
            else:
                # 檢測到靜音，計數增加
                self._silence_frames += 1
                if self._silence_frames >= self.silence_frames_threshold:
                    # 評估 KWS 結果：匹配成功則轉 Command，否則回 Idle
                    if self._kws_stream:
                        triggered = self._finalize_keyword_detection()

                    if triggered:
                        self._set_state(TriggerState.COMMAND)
                        # 發佈喚醒事件
                        self.triggered_pub.publish(Bool(data=True))
                        self.get_logger().info(f"✓ 喚醒觸發: '{self.kws_keyword}'")
                    else:
                        # KWS 未匹配，回到 Idle
                        self._set_state(TriggerState.IDLE)
                        self.get_logger().info("KWS 未匹配，重置為 IDLE")

        # Command 階段
        with self._lock:
            current_state = self._state

        if current_state == TriggerState.COMMAND:
            if self._in_command_initial_wait:
                # 初始等待期：累積幀數直到超過初始等待時長
                self._command_wait_frames += 1
                self._publish_audio(audio)
                if self._command_wait_frames >= self.command_initial_wait_frames:
                    # 初始等待期結束，開始監測說話結束
                    self._in_command_initial_wait = False
                    self._silence_frames = 0
                    self.get_logger().debug(
                        f"[Command] 初始等待期結束 ({self._command_wait_frames} 幀)，"
                        f"開始監測語音結束 (靜音閾值: {self.silence_frames_threshold} 幀)"
                    )
            else:
                # 監測說話結束
                if speech_detected:
                    self._silence_frames = 0
                    self._publish_audio(audio)
                    self.get_logger().debug("[Command] 偵測到語音，靜音計數重置")
                else:
                    self._silence_frames += 1
                    if self._silence_frames >= self.silence_frames_threshold:
                        # 語音結束，發佈最終塊
                        self._publish_audio(audio, is_final=True)
                        # 重置回 IDLE 狀態，供下一次喚醒
                        self._set_state(TriggerState.IDLE)
                        self.get_logger().info("VAD 檢測說話結束，重置為 IDLE")
                    else:
                        self._publish_audio(audio)

        return triggered

    def _accumulate_kws_audio(self, audio: np.ndarray) -> None:
        """在 KWS 流中累積音訊數據

        將音訊數據添加到 KWS 流並進行解碼

        Args:
            audio: 音訊數據 (float32 numpy array)
        """
        try:
            if not self.kws:
                return

            # 建立或使用持久化流
            if self._kws_stream is None:
                self._kws_stream = self.kws.create_stream()
                self.get_logger().debug("[Spotting] 建立新的 KWS 流")

            # 添加音訊到流
            if self._kws_stream is not None:
                self._kws_stream.accept_waveform(self.sample_rate, audio)

                # 處理已準備好的結果
                while self.kws.is_ready(self._kws_stream):
                    self.kws.decode_stream(self._kws_stream)

        except Exception as e:
            self.get_logger().error(f"✗ KWS 檢查失敗: {e}")
            self._kws_stream = None

    def _finalize_keyword_detection(self) -> bool:
        """評估 KWS 識別結果

        獲取 KWS 流的最終識別結果，並與預期關鍵字比對

        Returns:
            bool: 是否匹配成功
        """
        try:
            if not self.kws or self._kws_stream is None:
                return False

            # 獲取最終識別結果
            result = self.kws.get_result(self._kws_stream)

            if isinstance(result, str):
                detected_text = result.strip()
            else:
                detected_text = getattr(result, "text", None)
                if detected_text:
                    detected_text = detected_text.strip()

            self.get_logger().info(f"[Spotting 評估] KWS 識別結果: '{detected_text}'")
            self.get_logger().info(f"[期望關鍵字] '{self.kws_keyword.lower()}'")

            triggered = False

            if detected_text:
                detected_text_lower = detected_text.lower()

                # 移除聲調符號
                try:
                    nfd_form = unicodedata.normalize("NFD", detected_text_lower)
                    detected_text_no_tone = "".join(char for char in nfd_form if unicodedata.category(char) != "Mn")
                    self.get_logger().info(f"[正規化後] '{detected_text_no_tone}'")

                    # 匹配檢查
                    if self.kws_keyword.lower() in detected_text_no_tone:
                        self.get_logger().info(f"✓ Spotting 評估成功！")
                        triggered = True
                    else:
                        self.get_logger().debug(f"✗ Spotting 評估失敗: '{detected_text}' -> '{detected_text_no_tone}'")
                except Exception as e:
                    self.get_logger().debug(f"✗ 聲調移除失敗: {e}，放棄此次匹配")
            else:
                self.get_logger().debug("✗ KWS 識別結果為空")

            # 重置流以便下一次檢測
            self._kws_stream = None

            return triggered

        except Exception as e:
            self.get_logger().error(f"✗ KWS 最終評估失敗: {e}")
            self._kws_stream = None
            return False

    def _convert_keyword_to_phonemes(self, keyword: str, phone_dict: Dict[str, str]) -> str:
        """將關鍵字轉換為音素序列

        支持中文自動轉拼音，英文查表轉音素

        Args:
            keyword: 中文或英文關鍵字
            phone_dict: 英文單詞到音素的映射表

        Returns:
            str: 無空格的音素序列，無法轉換時回傳空字串
        """
        try:
            has_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in keyword)

            if has_chinese:
                pinyin_result = pinyin(keyword, style=Style.NORMAL)
                pinyin_list = [py[0] for py in pinyin_result if py]
                return "".join(pinyin_list)
            else:
                keyword_upper = keyword.upper()
                if keyword_upper in phone_dict:
                    return phone_dict[keyword_upper].replace(" ", "")
                else:
                    phonemes = []
                    for char in keyword_upper:
                        if char in phone_dict:
                            phonemes.append(phone_dict[char])
                    if phonemes:
                        return "".join(phonemes)
                    else:
                        return ""
        except Exception as e:
            self.get_logger().error(f"✗ 關鍵字轉換失敗: {e}")
            return ""

    def _load_english_phone_dict(self, kws_model_dir: Path) -> Dict[str, str]:
        """從 en.phone 檔案載入英文音素映射表

        Args:
            kws_model_dir: KWS 模型目錄

        Returns:
            Dict[str, str]: 英文單詞到音素的映射表 {word: phonemes}，載入失敗時回傳空字典
        """
        en_phone_file = kws_model_dir / "en.phone"
        phone_dict: Dict[str, str] = {}

        if en_phone_file.exists():
            try:
                with open(en_phone_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) > 1:
                                word = parts[0].upper()
                                phonemes = " ".join(parts[1:])
                                phone_dict[word] = phonemes
                return phone_dict
            except Exception as e:
                self.get_logger().error(f"✗ 載入英文音素表失敗: {e}")
                return {}

        return {}


def main(args: Optional[list] = None) -> None:
    """語音喚醒觸發節點進入點"""
    rclpy.init(args=args)
    node = VoiceTriggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_recorder()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
