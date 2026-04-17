#include <Arduino.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// 사용자 정의 데이터 및 모델
#include "dtln_model_data_001.h"

#include "dtln_inout_data_001.h"


// TFLM 관련 전역 변수
namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;

    // ESP32-S3의 넉넉한 RAM을 활용하여 할당 (모델 크기에 따라 조정)
    constexpr int kTensorArenaSize = 64 * 1024;
    alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

void T10_init() {


    Serial.println("--- DTLN Noise Suppression on ESP32-S3 ---");

    // 1. 모델 로드
    model = tflite::GetModel(g_dtln_noise_suppression_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Model schema version %d not supported!\n", model->version());
        return;
    }

    // 2. Op Resolver 설정 (LSTM 포함)
    // DTLN 모델에 필요한 핵심 연산자들 등록
    static tflite::MicroMutableOpResolver<3> micro_op_resolver;
    micro_op_resolver.AddUnidirectionalSequenceLSTM();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddLogistic();

    // 3. 인터프리터 초기화
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // 4. 텐서 메모리 할당
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("AllocateTensors() failed");
        return;
    }

    // 5. 입출력 텐서 포인터 획득
    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("TFLM Initialization Complete.");

    // 테스트 실행
    run_dtln_inference();
}

void run_dtln_inference() {
    Serial.println("Starting Inference Test...");

    // 입력 데이터 복사 (feature_data -> input tensor)
    for (size_t i = 0; i < input->bytes; ++i) {
        input->data.int8[i] = feature_data[i];
    }

    // 추론 수행 및 시간 측정
    uint32_t start_time = millis();
    TfLiteStatus invoke_status = interpreter->Invoke();
    uint32_t end_time = millis();

    if (invoke_status != kTfLiteOk) {
        Serial.println("Invoke failed!");
        return;
    }

    Serial.printf("Inference done in %d ms\n", end_time - start_time);

    // 결과 검증 (Golden Ref와 비교)
    int output_size = output->dims->data[2]; // 257
    int error_count = 0;

    for (int i = 0; i < output_size; i++) {
        if (output->data.int8[i] != golden_ref[i]) {
            error_count++;
        }
    }

    if (error_count == 0) {
        Serial.println("SUCCESS: Output matches Golden Reference!");
    } else {
        Serial.printf("FAILED: %d mismatches found.\n", error_count);
    }
}

void T10_run() {
    // 실시간 처리를 원할 경우 여기에 I2S 로직 추가
    delay(1000);
}

