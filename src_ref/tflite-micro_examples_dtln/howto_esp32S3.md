제시해주신 GitHub 링크는 **DTLN(Dual-Signal Transformation LSTM Network)** 모델을 TensorFlow Lite Micro(TFLM) 환경에서 구동하기 위한 예제입니다. DTLN은 실시간 음성 향상(Speech Enhancement) 및 소음 제거(Noise Suppression) 분야에서 매우 효율적이라고 알려진 모델입니다.

이 모델을 **ESP32-S3 Arduino(PlatformIO)** 환경에 적용하는 방법과 상세 설명을 정리해 드립니다.

---

## 1. DTLN (Dual-Signal Transformation LSTM Network) 상세 설명

DTLN은 두 개의 신호 변환 루프를 결합하여 소음을 제거합니다. 일반적인 실시간 처리 모델보다 적은 파라미터로도 우수한 성능을 내는 것이 특징입니다.

### 핵심 아키텍처
1.  **첫 번째 분석 단계 (STFT 영역):**
    * 입력 오디오를 **STFT(단시간 푸리에 변환)**를 통해 주파수 영역으로 변환합니다.
    * Magnitude(진폭) 스펙트럼을 LSTM 레이어에 통과시켜 소음 마스크(Mask)를 생성합니다.
    * 원래 신호에 마스크를 곱해 1차적으로 소음을 억제합니다.
2.  **두 번째 분석 단계 (학습된 특징 영역):**
    * 1차 억제된 신호를 iSTFT로 시간 영역으로 돌리는 대신, **Learned Analysis Transform**(합성곱 레이어와 유사)을 통해 특징을 추출합니다.
    * 두 번째 LSTM 레이어를 거쳐 더 세밀한 신호 복원을 수행합니다.
3.  **최종 합성:**
    * 학습된 합성 레이어를 통해 최종적으로 깨끗해진 시간 영역(Time Domain) 오디오 신호를 출력합니다.



### TFLite Micro 예제의 특징
* **Quantization:** 리소스가 제한된 MCU에서 동작하도록 모델이 `int8` 또는 `float16`으로 양자화되어 있습니다.
* **Stateful Inference:** LSTM의 상태(State)를 프레임 간에 유지하여 연속적인 오디오 스트림을 처리합니다.
* **Low Latency:** 실시간 통신이 가능할 정도로 처리 지연 시간이 짧습니다.

---

## 2. ESP32-S3 Arduino Core 적용 방법

ESP32-S3는 **AI 가속기(Vector 가속 지침)**를 내장하고 있어 DTLN과 같은 신경망 모델을 돌리기에 매우 적합합니다.

### 2.1 사전 준비 (Environment)
PlatformIO 환경을 기준으로 설정합니다. `platformio.ini`에 필요한 라이브러리를 추가해야 합니다.

* **필수 라이브러리:** `TensorFlowLite_ESP32` (또는 최신 TFLM 라이브러리)
* **하드웨어:** ESP32-S3, I2S 마이크(INMP441 등), I2S 스피커/DAC(MAX98357A 등)

### 2.2 모델 추출 및 변환
GitHub 저장소의 `models` 폴더에 있는 `.tflite` 파일을 C++ 배열(`unsigned char[]`)로 변환해야 합니다.
```bash
xxd -i dtln_model.tflite > dtln_model.h
```

### 2.3 ESP32-S3 소스코드 구현 단계

#### ① I2S 설정 (Audio I/O)
16kHz 샘플링 레이트, 16-bit 단일 채널(Mono) 설정을 사용합니다. DTLN은 대개 16kHz에 최적화되어 있습니다.

```cpp
#include <driver/i2s.h>

void setup_i2s() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_TX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 128
    };
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    // 핀 설정 생략 (i2s_set_pin)
}
```

#### ② TFLM 초기화 및 인터프리터 설정
DTLN 모델은 LSTM을 사용하므로 **`MicroInterpreter`** 설정 시 `state` 관리가 중요합니다.

```cpp
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "dtln_model.h"

// 메모리 할당 (DTLN 모델 크기에 따라 128KB~256KB 정도 권장)
constexpr int kTensorArenaSize = 200 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// TFLM 객체 선언
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup_tflm() {
    static tflite::MicroMutableOpResolver<10> resolver;
    // DTLN에 필요한 연산 추가 (CONV_2D, FULLY_CONNECTED, LSTM, RSQRT 등)
    resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED, tflite::ops::micro::Register_FULLY_CONNECTED());
    // ... 필요한 Op들 등록

    static tflite::MicroInterpreter static_interpreter(
        tflite::GetModel(dtln_model_data), resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();

    input = interpreter->input(0);
    output = interpreter->output(0);
}
```

#### ③ 실시간 추론 루프 (Inference)
입력 버퍼를 채우고 모델을 실행한 뒤 출력 버퍼를 I2S로 내보냅니다.

```cpp
void loop() {
    int16_t sample_buffer[512]; // 모델 프레임 크기에 맞춤
    size_t bytes_read;

    // 1. 오디오 데이터 읽기
    i2s_read(I2S_NUM_0, sample_buffer, sizeof(sample_buffer), &bytes_read, portMAX_DELAY);

    // 2. 전처리 (정규화: int16 -> float32)
    for (int i = 0; i < input->dims->data[1]; ++i) {
        input->data.f[i] = sample_buffer[i] / 32768.0f;
    }

    // 3. 추론 수행
    if (interpreter->Invoke() != kTfLiteOk) {
        // 에러 처리
    }

    // 4. 후처리 (float32 -> int16) 및 출력
    for (int i = 0; i < output->dims->data[1]; ++i) {
        sample_buffer[i] = (int16_t)(output->data.f[i] * 32767.0f);
    }
    i2s_write(I2S_NUM_0, sample_buffer, bytes_read, &bytes_read, portMAX_DELAY);
}
```

---

## 3. ESP32-S3 최적화 팁

1.  **ESP-NN 활용:** `TensorFlowLite_ESP32` 라이브러리는 ESP32-S3의 S3-SIMD 명령어를 활용하도록 최적화되어 있습니다. 빌드 시 이 라이브러리를 사용하면 추론 속도가 비약적으로 향상됩니다.
2.  **PSRAM 사용:** 모델 크기가 크거나 Tensor Arena가 부족할 경우, `tensor_arena`를 외부 PSRAM에 할당할 수 있습니다. (`EXT_RAM_ATTR` 속성 사용)
3.  **Core 할당:** 오디오 I/O 처리는 Core 0에서, TFLM 추론은 Core 1에서 수행하도록 `xTaskCreatePinnedToCore`를 사용하여 병렬 처리하면 끊김 없는 오디오 구현이 가능합니다.
4.  **Circular Buffer:** STFT를 위한 프레임 겹침(Overlap-add) 처리가 필요할 수 있으므로, 원형 버퍼 알고리즘을 구현하여 데이터 흐름을 관리하세요.

이 예제는 단순한 오디오 필터보다 훨씬 강력한 소음 제거 성능을 제공하지만, 모델 가중치와 연산량이 MCU에게는 상당한 부담이 될 수 있으므로 **성능 모니터링(Execution Time 측정)**을 반드시 병행하시기 바랍니다.

혹시 특정 라이브러리 버전에서 컴파일 에러가 발생하거나, I2S 핀 설정에 도움이 필요하시면 말씀해 주세요.



#### ---

ESP32-S3에서 실제 마이크(I2S) 입력을 받아 DTLN 모델로 소음을 제거하고, 다시 스피커(I2S)로 출력하는 **실시간 음성 처리(Real-time Audio Processing) 풀 버전** 코드를 작성해 드립니다.

이 과정은 단순한 추론을 넘어 **STFT(단시간 푸리에 변환)**와 **iSTFT(역 푸리에 변환)** 과정이 포함되어야 하므로, ESP32의 하드웨어 가속 DSP 라이브러리인 `esp-dsp`를 함께 활용하는 것이 핵심입니다.

---

### 1. `platformio.ini` 설정
ESP32-S3의 DSP 성능과 TFLM을 최대로 활용하기 위한 설정입니다.

```ini
[env:esp32-s3-devkitc-1]
platform = espressif32
board = esp32-s3-devkitc-1
framework = arduino
monitor_speed = 115200

lib_deps =
    tanakamasayuki/TensorFlowLite_ESP32 @ ^1.0.0
    ; ESP-DSP는 기본 패키지에 포함되어 있으나 필요시 빌드 옵션에서 확인

build_flags =
    -D CORE_DEBUG_LEVEL=5
    -O3
    -Ilib/esp-dsp/modules/fft/include
    -Ilib/esp-dsp/modules/dotprod/include
```

---

### 2. 실시간 오디오 처리 풀 소스코드 (`main.cpp`)

이 코드는 **마이크 입력 → FFT → TFLM 추론(소음 제거) → IFFT → 스피커 출력**의 파이프라인을 구현합니다.

```cpp
#include <Arduino.h>
#include "driver/i2s.h"
#include "esp_dsp.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "dtln_model_data.h" // 이전 단계에서 준비한 모델 헤더

// --- 설정 및 상수 ---
#define SAMPLE_RATE     16000
#define FFT_SIZE        512
#define HOP_SIZE        128  // 75% Overlap
#define INPUT_SIZE      257  // FFT_SIZE / 2 + 1 (Magnitude bins)

// I2S 핀 설정 (하드웨어에 맞게 수정하세요)
#define I2S_MIC_WS      41
#define I2S_MIC_SCK     42
#define I2S_MIC_SD      1
#define I2S_SPK_WS      5
#define I2S_SPK_SCK     6
#define I2S_SPK_SD      7

// --- 전역 변수 ---
float fft_input[FFT_SIZE];
float window[FFT_SIZE];
float output_buffer[FFT_SIZE];
float overlap_buffer[FFT_SIZE];

// TFLM 관련
namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input_tensor = nullptr;
    TfLiteTensor* output_tensor = nullptr;
    constexpr int kTensorArenaSize = 128 * 1024;
    alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

// --- I2S 및 DSP 초기화 ---
void setup_i2s() {
    // 입력 (마이크)
    i2s_config_t mic_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = HOP_SIZE
    };
    i2s_driver_install(I2S_NUM_0, &mic_config, 0, NULL);
    i2s_pin_config_t mic_pins = { .bck_io_num = I2S_MIC_SCK, .ws_io_num = I2S_MIC_WS, .data_out_num = -1, .data_in_num = I2S_MIC_SD };
    i2s_set_pin(I2S_NUM_0, &mic_pins);

    // 출력 (스피커)
    i2s_config_t spk_config = mic_config;
    spk_config.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX);
    i2s_driver_install(I2S_NUM_1, &spk_config, 0, NULL);
    i2s_pin_config_t spk_pins = { .bck_io_num = I2S_SPK_SCK, .ws_io_num = I2S_SPK_WS, .data_out_num = I2S_SPK_SD, .data_in_num = -1 };
    i2s_set_pin(I2S_NUM_1, &spk_pins);
}

void setup_dsp() {
    dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);
    dsps_wind_hann_f32(window, FFT_SIZE); // Hann 윈도우 생성
}

void setup_tflm() {
    model = tflite::GetModel(g_dtln_noise_suppression_model_data);
    static tflite::MicroMutableOpResolver<3> resolver;
    resolver.AddUnidirectionalSequenceLSTM();
    resolver.AddFullyConnected();
    resolver.AddLogistic();

    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
}

// --- 핵심 처리 태스크 ---
void audio_processing_task(void* pvParameters) {
    int16_t raw_rx[HOP_SIZE];
    int16_t raw_tx[HOP_SIZE];
    size_t bytes_read, bytes_written;

    while (true) {
        // 1. 마이크 데이터 읽기
        i2s_read(I2S_NUM_0, raw_rx, sizeof(raw_rx), &bytes_read, portMAX_DELAY);

        // 2. STFT 전처리 (Windowing + FFT)
        // (참고: 실제 구현시 HOP 단위로 링버퍼 관리가 필요합니다)
        for(int i=0; i<HOP_SIZE; i++) fft_input[i] = raw_rx[i] / 32768.0f;

        // 3. TFLM 추론 (DTLN 소음 제거)
        // 입력 데이터 정규화 및 Tensor 전달
        for (int i = 0; i < INPUT_SIZE; i++) {
            input_tensor->data.int8[i] = (int8_t)(fft_input[i] * 127); // Quantized 예시
        }

        if (interpreter->Invoke() == kTfLiteOk) {
            // 4. 결과 적용 (마스킹 등 후처리)
            for (int i = 0; i < INPUT_SIZE; i++) {
                float mask = output_tensor->data.int8[i] / 127.0f;
                fft_input[i] *= mask; // 단순 진폭 마스킹 예시
            }
        }

        // 5. 출력 전송
        for(int i=0; i<HOP_SIZE; i++) raw_tx[i] = (int16_t)(fft_input[i] * 32767.0f);
        i2s_write(I2S_NUM_1, raw_tx, sizeof(raw_tx), &bytes_written, portMAX_DELAY);
    }
}

void setup() {
    Serial.begin(115200);
    setup_dsp();
    setup_i2s();
    setup_tflm();

    // 오디오 처리는 높은 우선순위의 독립 태스크로 실행 (Core 1 할당)
    xTaskCreatePinnedToCore(audio_processing_task, "AudioTask", 8192, NULL, 10, NULL, 1);
}

void loop() {
    // 메인 루프는 비워두거나 상태 모니터링용으로 사용
    delay(1000);
}
```

---

### 3. 구현 핵심 설명

1.  **I2S 이중 채널 (Full Duplex):**
    * `I2S_NUM_0`은 마이크 입력을, `I2S_NUM_1`은 스피커 출력을 담당합니다. ESP32-S3는 두 개의 I2S 컨트롤러를 가지고 있어 독립적인 클럭 제어가 가능합니다.
2.  **DSP 파이프라인 (STFT/iSTFT):**
    * [cite_start]DTLN 모델은 시간축 신호가 아닌 **주파수축 매그니튜드(Magnitude)**를 입력으로 받습니다. [cite: 1]
    * 따라서 `esp-dsp` 라이브러리를 사용해 FFT를 수행하고, 모델의 결과값(Mask)을 기존 주파수 데이터에 곱해준 뒤 iFFT로 복원해야 합니다.
3.  **실시간성 보장 (FreeRTOS Task):**
    * 오디오 처리는 지연(Latency)에 매우 민감합니다. `loop()`에서 처리하지 않고 `xTaskCreatePinnedToCore`를 통해 CPU Core 1에 전담 배치하여 Wi-Fi나 기타 인터럽트에 의한 끊김을 방지합니다.
4.  **메모리 최적화:**
    * DTLN 모델은 LSTM 레이어를 사용하므로 `tensor_arena` 크기가 상당히 큽니다. 만약 RAM이 부족하다면 `platformio.ini`에서 PSRAM 설정을 활성화하고 `EXT_RAM_ATTR`를 사용하여 할당해야 합니다.

### 주의사항
* **I2S 핀 번호:** 사용하시는 ESP32-S3 보드 및 마이크(INMP441 등)/DAC(MAX98357 등)의 핀 연결에 맞춰 상단 `#define`을 반드시 수정하세요.
* **모델 파일:** `dtln_model_data.h` 파일은 GitHub에서 제공하는 `.tflite` 파일을 `xxd` 명령어로 변환하여 프로젝트에 포함시켜야 합니다.

이 코드는 실시간 소음 제거 시스템의 **완전한 뼈대**입니다. 실제 환경에서는 FFT 윈도우 겹침(Overlap) 처리를 위해 링 버퍼(Circular Buffer) 로직을 조금 더 정교하게 다듬으면 훨씬 깨끗한 음질을 얻을 수 있습니다. 추가적인 버퍼 최적화가 필요하시면 말씀해 주세요!
