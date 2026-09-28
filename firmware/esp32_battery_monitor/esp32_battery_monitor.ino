/*
 * SmartCell - ESP32 battery voltage/current logger (INA219)
 *
 * Wiring: battery+ -> INA219 VIN+, INA219 VIN- -> load(+),
 *         load(-) -> battery- (shared ground with ESP32).
 *         INA219 SDA/SCL -> ESP32 default I2C pins (GPIO21/GPIO22).
 * This is high-side sensing: current is positive while the battery is
 * DISCHARGING into the load. If your reading comes out negative, either
 * swap VIN+/VIN- or negate current in the Python script (one flag, see
 * CURRENT_SIGN in soh_rul_monitor.py) - don't rewire for it.
 *
 * Output: one CSV line per sample over Serial (115200 baud):
 *   millis,voltage_V,current_A
 *
 * Requires the Adafruit INA219 library (Library Manager -> "Adafruit INA219").
 */

#include <Wire.h>
#include <Adafruit_INA219.h>

Adafruit_INA219 ina219;

const unsigned long SAMPLE_INTERVAL_MS = 1000;  // 1 Hz - plenty for a battery that changes over minutes/hours
unsigned long lastSample = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  Wire.begin();
  if (!ina219.begin()) {
    Serial.println("ERROR: INA219 not found - check wiring/I2C address");
    while (1) { delay(1000); }
  }

  // Uncomment if you're monitoring near the sensor's max current for your
  // shunt (default range is +-3.2A at 0.1 ohm shunt on most breakouts):
  // ina219.setCalibration_32V_1A();

  Serial.println("millis,voltage_V,current_A");  // header line, Python skips non-numeric lines
}

void loop() {
  unsigned long now = millis();
  if (now - lastSample < SAMPLE_INTERVAL_MS) return;
  lastSample = now;

  float busVoltage_V = ina219.getBusVoltage_V();
  float shuntVoltage_mV = ina219.getShuntVoltage_mV();
  float current_A = ina219.getCurrent_mA() / 1000.0;
  float terminalVoltage_V = busVoltage_V + (shuntVoltage_mV / 1000.0);  // battery-side terminal voltage

  Serial.print(now);
  Serial.print(",");
  Serial.print(terminalVoltage_V, 3);
  Serial.print(",");
  Serial.println(current_A, 4);
}
