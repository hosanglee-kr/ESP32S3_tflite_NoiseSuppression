#include <Arduino.h>

// The build version comes from an environment variable. Use the VERSION
// define wherever the version is needed.
#define STRINGIZER(arg) #arg
#define STR_VALUE(arg)	STRINGIZER(arg)
#define VERSION			STR_VALUE(BUILD_VERSION)

#define S10
#ifdef S10
	#include "S10_streams-i2s-webserver_wav_001.h"
#endif

void setup() {
	// delay(5000);

	Serial.begin(115200);
	Serial.print("Version: ");
	Serial.println(VERSION);

#ifdef S10
	S10_setup();
#endif

	Serial.println("11111");
}

void loop() {
#ifdef S10
	S10_loop();
#endif
}
