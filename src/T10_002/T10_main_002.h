

#include <Arduino.h>


#include "driver/i2s.h"		// 레거시 드라이버


#include "esp_dsp.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "dtln_model_data_002.h"  // 이전 단계에서 준비한 모델 헤더

// extern const unsigned char g_dtln_noise_suppression_model_data[];
// extern const int g_dtln_noise_suppression_model_data_len;

extern unsigned char dtln_noise_suppression_tflite[];
extern unsigned int dtln_noise_suppression_tflite_len;

// --- 설정 및 상수 ---
#define 	SAMPLE_RATE 16000
#define 	FFT_SIZE	512
#define 	HOP_SIZE	128	 // 75% Overlap
#define 	INPUT_SIZE	257	 // FFT_SIZE / 2 + 1 (Magnitude bins)

// I2S 핀 설정 (하드웨어에 맞게 수정하세요)
#define I2S_MIC_WS	41
#define I2S_MIC_SCK 42
#define I2S_MIC_SD	1
#define I2S_SPK_WS	5
#define I2S_SPK_SCK 6
#define I2S_SPK_SD	7

// --- 전역 변수 ---
float fft_input[FFT_SIZE];
float window[FFT_SIZE];
float output_buffer[FFT_SIZE];
float overlap_buffer[FFT_SIZE];

// TFLM 관련
namespace {
const tflite::Model*	  model			   = nullptr;
tflite::MicroInterpreter* interpreter	   = nullptr;
TfLiteTensor*			  input_tensor	   = nullptr;
TfLiteTensor*			  output_tensor	   = nullptr;
constexpr int			  kTensorArenaSize = 128 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// --- I2S 및 DSP 초기화 ---
void setup_i2s() {
	// 입력 (마이크)
	i2s_config_t mic_config = {
		.mode				  = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
		.sample_rate		  = SAMPLE_RATE,
		.bits_per_sample	  = I2S_BITS_PER_SAMPLE_16BIT,
		.channel_format		  = I2S_CHANNEL_FMT_ONLY_LEFT,
		.communication_format = I2S_COMM_FORMAT_STAND_I2S,
		.intr_alloc_flags	  = ESP_INTR_FLAG_LEVEL1,
		.dma_buf_count		  = 4,
		.dma_buf_len		  = HOP_SIZE};
	i2s_driver_install(I2S_NUM_0, &mic_config, 0, NULL);
	i2s_pin_config_t mic_pins = {.bck_io_num = I2S_MIC_SCK, .ws_io_num = I2S_MIC_WS, .data_out_num = -1, .data_in_num = I2S_MIC_SD};
	i2s_set_pin(I2S_NUM_0, &mic_pins);

	// 출력 (스피커)
	i2s_config_t spk_config = mic_config;
	spk_config.mode			= (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX);
	i2s_driver_install(I2S_NUM_1, &spk_config, 0, NULL);
	i2s_pin_config_t spk_pins = {.bck_io_num = I2S_SPK_SCK, .ws_io_num = I2S_SPK_WS, .data_out_num = I2S_SPK_SD, .data_in_num = -1};
	i2s_set_pin(I2S_NUM_1, &spk_pins);
}

void setup_dsp() {
	dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);
	dsps_wind_hann_f32(window, FFT_SIZE);  // Hann 윈도우 생성
}

void setup_tflm() {
	// 1. 에러 리포터 선언 (static으로 선언하여 메모리 유지)
    //static tflite::ErrorReporter micro_error_reporter;
	static tflite::MicroErrorReporter micro_error_reporter;

	// 2. 모델 로드 및 Resolver 설정 (기존과 동일)
	model = tflite::GetModel(dtln_noise_suppression_tflite);
	static tflite::MicroMutableOpResolver<3> resolver;
	resolver.AddUnidirectionalSequenceLSTM();

	resolver.AddFullyConnected();
	resolver.AddLogistic();

	// 3. 인터프리터 생성 (마지막 인자로 &micro_error_reporter 추가)
    static tflite::MicroInterpreter static_interpreter(
		model,
		resolver,
		tensor_arena,
		kTensorArenaSize,
		&micro_error_reporter // 이 부분이 빠져서 에러가 발생했습니다.
    );

	interpreter = &static_interpreter;
	interpreter->AllocateTensors();
	input_tensor  = interpreter->input(0);
	output_tensor = interpreter->output(0);
}

// --- 핵심 처리 태스크 ---
void audio_processing_task(void* pvParameters) {
	int16_t raw_rx[HOP_SIZE];
	int16_t raw_tx[HOP_SIZE];
	size_t	bytes_read, bytes_written;

	while (true) {
		// 1. 마이크 데이터 읽기
		i2s_read(I2S_NUM_0, raw_rx, sizeof(raw_rx), &bytes_read, portMAX_DELAY);

		// 2. STFT 전처리 (Windowing + FFT)
		// (참고: 실제 구현시 HOP 단위로 링버퍼 관리가 필요합니다)
		for (int i = 0; i < HOP_SIZE; i++) fft_input[i] = raw_rx[i] / 32768.0f;

		// 3. TFLM 추론 (DTLN 소음 제거)
		// 입력 데이터 정규화 및 Tensor 전달
		for (int i = 0; i < INPUT_SIZE; i++) {
			input_tensor->data.int8[i] = (int8_t)(fft_input[i] * 127);	// Quantized 예시
		}

		if (interpreter->Invoke() == kTfLiteOk) {
			// 4. 결과 적용 (마스킹 등 후처리)
			for (int i = 0; i < INPUT_SIZE; i++) {
				float mask = output_tensor->data.int8[i] / 127.0f;
				fft_input[i] *= mask;  // 단순 진폭 마스킹 예시
			}
		}

		// 5. 출력 전송
		for (int i = 0; i < HOP_SIZE; i++) raw_tx[i] = (int16_t)(fft_input[i] * 32767.0f);
		i2s_write(I2S_NUM_1, raw_tx, sizeof(raw_tx), &bytes_written, portMAX_DELAY);
	}
}

void T10_init() {
	setup_dsp();
	setup_i2s();
	setup_tflm();

	// 오디오 처리는 높은 우선순위의 독립 태스크로 실행 (Core 1 할당)
	xTaskCreatePinnedToCore(audio_processing_task, "AudioTask", 8192, NULL, 10, NULL, 1);
}

void T10_run() {
	// 메인 루프는 비워두거나 상태 모니터링용으로 사용
	delay(1000);
}
