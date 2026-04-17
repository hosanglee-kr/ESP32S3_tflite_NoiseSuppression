
#ifndef DTLN_MODEL_DATA_H_
#define DTLN_MODEL_DATA_H_

// GitHub에서 다운로드한 .tflite 파일을 xxd -i 명령어로 변환하여 여기에 붙여넣으세요.
// xxd -i dtln_noise_suppression.tflite > dtln_model_data_001.h

extern const unsigned char g_dtln_noise_suppression_model_data[];
extern const int g_dtln_noise_suppression_model_data_len;

#endif

