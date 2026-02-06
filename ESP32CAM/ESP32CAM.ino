#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// ================= WIFI =================
const char* ssid = "REDMI Note 15 Pro 5G";
const char* password = "12345678";

// ================= SERVER =================
const char* SERVER_URL = "http://10.76.11.184:5000/ocr";
const char* SERVER_HEALTH_URL = "http://10.76.11.184:5000/";

// ================= SERVER CHECK =================
bool checkServer() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected; cannot check server");
    return false;
  }

  HTTPClient http;
  http.setTimeout(5000);
  http.begin(SERVER_HEALTH_URL);
  int httpCode = http.GET();

  Serial.print("Server GET status: ");
  Serial.println(httpCode);
  if (httpCode > 0) {
    String payload = http.getString();
    if (payload.length()) {
      Serial.println(payload);
    }
  } else {
    Serial.printf("Server GET error: %s\n", http.errorToString(httpCode).c_str());
  }

  http.end();
  return (httpCode == 200);
}

// ================= PINS =================
#define FLASH_PIN 4   // ESP32-CAM onboard flash LED

// ================= FLASH CONTROL =================
void flashOn() {
  digitalWrite(FLASH_PIN, HIGH);
}

void flashOff() {
  digitalWrite(FLASH_PIN, LOW);
}

// ================= CAMERA INIT =================
void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = 5;
  config.pin_d1       = 18;
  config.pin_d2       = 19;
  config.pin_d3       = 21;
  config.pin_d4       = 36;
  config.pin_d5       = 39;
  config.pin_d6       = 34;
  config.pin_d7       = 35;
  config.pin_xclk     = 0;
  config.pin_pclk     = 22;
  config.pin_vsync    = 25;
  config.pin_href     = 23;
  config.pin_sscb_sda = 26;
  config.pin_sscb_scl = 27;
  config.pin_pwdn     = 32;
  config.pin_reset   = -1;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // config.frame_size   = FRAMESIZE_VGA;
  // config.jpeg_quality = 12;
  // config.fb_count     = 1;
  config.frame_size   = FRAMESIZE_XGA;
  config.jpeg_quality = 10;
  config.fb_count     = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    while (true);
  }
}

// ================= SEND IMAGE =================
void takeAndSendImage() {
  // 1. Capture
  flashOn();
  delay(500); // Give sensor time to adjust to light
  
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    flashOff();
    return;
  }

  // 2. Prepare Connection
  HTTPClient http;
  http.begin(SERVER_URL);
  http.setTimeout(30000); // OCR + LLM + TTS takes time, increase timeout
  http.addHeader("Content-Type", "image/jpeg");

  Serial.printf("Sending %u bytes to %s\n", fb->len, SERVER_URL);
  
  // 3. POST raw buffer
  int httpCode = http.POST(fb->buf, fb->len);

  if (httpCode > 0) {
    Serial.printf("Response: %d\n", httpCode);
    String payload = http.getString();
    Serial.println(payload);
  } else {
    Serial.printf("Error: %s\n", http.errorToString(httpCode).c_str());
  }

  http.end();
  esp_camera_fb_return(fb);
  flashOff();
}

// ================= SERIAL COMMAND =================
void handleSerialCommand() {
  if (!Serial.available()) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();

  if (cmd == "take") {
    Serial.println("Command received: take");
    takeAndSendImage();
  } else if (cmd == "check") {
    Serial.println("Command received: check");
    checkServer();
  }
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);

  pinMode(FLASH_PIN, OUTPUT);

  // Flash blink on power-up
  flashOn();
  delay(300);
  flashOff();

  Serial.println("Booting ESP32-CAM...");

  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");

  // Quick GET request to confirm backend is reachable.
  checkServer();

  initCamera();
  Serial.println("Camera ready");

  Serial.println("Type 'take' in Serial Monitor to capture image");
  Serial.println("Type 'check' in Serial Monitor to ping server");
}

// ================= LOOP =================
void loop() {
  handleSerialCommand();
}
