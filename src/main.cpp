#include <Arduino.h>



#define T10
#ifdef T10
	#include "T10_002/T10_main_002.h"
#endif

void setup() {


	Serial.begin(115200);

	#ifdef T10
		T10_init();
	#endif


}

void loop() {
	#ifdef T10
		T10_run();
	#endif
}
