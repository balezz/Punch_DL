
#include <pcf8563.h>
#include <TFT_eSPI.h> // Graphics and font library for ST7735 driver chip
#include <SPI.h>
#include <Wire.h>

#include <time.h>
#include "sensor.h"
#include "esp_adc_cal.h"

#include "BluetoothSerial.h"

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif

BluetoothSerial SerialBT;


#define TP_PIN_PIN          33
#define I2C_SDA_PIN         21
#define I2C_SCL_PIN         22
#define IMU_INT_PIN         38
#define RTC_INT_PIN         34
#define BATT_ADC_PIN        35
#define VBUS_PIN            36
#define TP_PWR_PIN          25
#define LED_PIN             4
#define CHARGE_PIN          32

#define DST_OFFSET          3600 //1 hour
#define TZ_OFFSET           -18000 //UTC-5
#define IDLE_TIMEOUT        30000 //30 seconds

#define NUM_FUNCS           2    //number of different functions to switch between when pressing button
#define IMU_SR              50  //IMU sample rate

extern MPU9250 IMU;

TFT_eSPI tft = TFT_eSPI();  // Invoke library, pins defined in User_Setup.h
PCF8563_Class rtc;
//WiFiManager wifiManager;

char buff[256];
char bt_buffer[64];

bool rtcIrq = false;
bool initial = 1;
bool otaStart = false;
bool otaSetup = false;
bool delayedSetup = false;
bool wifiConnected = false;
bool wifiStateChanged = false;

int bt_delay = (int)1000 / IMU_SR;
uint8_t func_select = 0;
uint8_t orig_mm = 99;
uint8_t xcolon = 0;
uint32_t targetTime = 0;       // for next 1 second timeout
uint32_t colour = 0;
uint8_t debugSelect = 0;
int vref = 1100;

bool pressed = false;
uint32_t pressedTime = 0;
bool charge_indication = false;

uint8_t hh, mm, ss ;


void scanI2Cdevice(void)
{
    uint8_t err, addr;
    int nDevices = 0;
    for (addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        err = Wire.endTransmission();
        if (err == 0) {
            Serial.print("I2C device found at address 0x");
            if (addr < 16)
                Serial.print("0");
            Serial.print(addr, HEX);
            Serial.println(" !");
            nDevices++;
        } else if (err == 4) {
            Serial.print("Unknow error at address 0x");
            if (addr < 16)
                Serial.print("0");
            Serial.println(addr, HEX);
        }
    }
    if (nDevices == 0)
        Serial.println("No I2C devices found\n");
    else
        Serial.println("Done\n");
}

void setupADC()
{
    esp_adc_cal_characteristics_t adc_chars;
    esp_adc_cal_value_t val_type = esp_adc_cal_characterize((adc_unit_t)ADC_UNIT_1, (adc_atten_t)ADC1_CHANNEL_6, (adc_bits_width_t)ADC_WIDTH_BIT_12, 1100, &adc_chars);
    //Check type of calibration value used to characterize ADC
    if (val_type == ESP_ADC_CAL_VAL_EFUSE_VREF) {
        Serial.printf("eFuse Vref:%u mV", adc_chars.vref);
        vref = adc_chars.vref;
    } else if (val_type == ESP_ADC_CAL_VAL_EFUSE_TP) {
        Serial.printf("Two Point --> coeff_a:%umV coeff_b:%umV\n", adc_chars.coeff_a, adc_chars.coeff_b);
    } else {
        Serial.println("Default Vref: 1100mV");
    }
}

void setupRTC()
{
    rtc.begin(Wire);
    //Check if the RTC clock matches, if not, use compile time
    //rtc.check();

    RTC_Date datetime = rtc.getDateTime();
    hh = datetime.hour;
    mm = datetime.minute;
    ss = datetime.second;
}

void setup(void)
{
    uint32_t start = millis();
    Serial.begin(19200);
    Serial.print("Start: ");
    Serial.println(start);
    
    //use bluetooth 
    SerialBT.begin("ESP32test"); //Bluetooth device name
    Serial.println("The device started, now you can pair it with bluetooth!");
    
    Serial.println(millis());

    tft.init();
    tft.setRotation(1);
    tft.setSwapBytes(true);
    //tft.pushImage(0, 0,  160, 80, ttgo);
    //tft.fillScreen(TFT_BLACK);
    Serial.print("TFT: ");
    Serial.println(millis());

    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    Wire.setClock(400000);

    Serial.print("Wire: ");
    Serial.println(millis());

    setupMPU9250(); //Start IMU
    Serial.print("MPU: ");
    Serial.println(millis());
    
    setupADC();
    Serial.print("ADC: ");
    Serial.println(millis());


    tft.fillScreen(TFT_BLACK);

    tft.setTextColor(TFT_YELLOW, TFT_BLACK); // Note: the new fonts do not draw the background colour

    //targetTime = millis() + 1000;

    pinMode(TP_PIN_PIN, INPUT);
    //! Must be set to pull-up output mode in order to wake up in deep sleep mode
    pinMode(TP_PWR_PIN, PULLUP);
    digitalWrite(TP_PWR_PIN, HIGH);

    pinMode(LED_PIN, OUTPUT);

    pinMode(CHARGE_PIN, INPUT_PULLUP);
    attachInterrupt(CHARGE_PIN, [] {
        charge_indication = true;
    }, CHANGE);

    if (digitalRead(CHARGE_PIN) == LOW) {
        charge_indication = true;
    }
    Serial.print("Setup Complete: ");
    Serial.println(millis());
    pressedTime = millis();
}

String getVoltage()
{
    uint16_t v = analogRead(BATT_ADC_PIN);
    float battery_voltage = ((float)v / 4095.0) * 2.0 * 3.3 * (vref / 1000.0);
    return String(battery_voltage) + "V";
}


void Write_BT(){
  uint32_t start;
  for(int i=0; i<IMU_SR; i++){
    start = millis();
    readMPU9250();
    snprintf(bt_buffer, sizeof(bt_buffer), "acc %.2f  %.2f  %.2f gyro %.2f  %.2f  %.2f \n", 
                          (int)1000 * IMU.ax, (int)1000 * IMU.ay, (int)1000 * IMU.az, 
                          IMU.gx, IMU.gy, IMU.gz);
    
    SerialBT.write((uint8_t *)bt_buffer, sizeof(bt_buffer));
    // Serial.println("Cycle read IMU - send BT takes: ");
    // Serial.print(millis() - start);
    delay(bt_delay);
  }
}

void Show_IMU()
{
    Write_BT();
    
    tft.setTextColor(TFT_GREEN, TFT_BLACK);
    tft.fillScreen(TFT_BLACK);
    tft.setTextDatum(TL_DATUM);

    snprintf(buff, sizeof(buff), "--  ACC  GYR   MAG");
    tft.drawString(buff, 0, 0);
    snprintf(buff, sizeof(buff), "x %.2f  %.2f  %.2f", (int)1000 * IMU.ax, IMU.gx, IMU.mx);
    tft.drawString(buff, 0, 16);
    snprintf(buff, sizeof(buff), "y %.2f  %.2f  %.2f", (int)1000 * IMU.ay, IMU.gy, IMU.my);
    tft.drawString(buff, 0, 32);
    snprintf(buff, sizeof(buff), "z %.2f  %.2f  %.2f", (int)1000 * IMU.az, IMU.gz, IMU.mz);
    tft.drawString(buff, 0, 48);

    
    // delay(200);
}

void Go_To_Sleep()
{
    //tft.setTextColor(TFT_GREEN, TFT_BLACK);
    //tft.setTextDatum(MC_DATUM);
    //tft.drawString("Press again to wake up",  tft.width() / 2, tft.height() / 2 );
    IMU.setSleepEnabled(true);
    Serial.println("Go to Sleep");
    delay(100);
    tft.writecommand(ST7735_SLPIN);
    tft.writecommand(ST7735_DISPOFF);
    esp_sleep_enable_ext1_wakeup(GPIO_SEL_33, ESP_EXT1_WAKEUP_ANY_HIGH);
    esp_deep_sleep_start();
}


void loop()
{
  if (digitalRead(TP_PIN_PIN) == HIGH) {
    func_select += 1;
    func_select %= NUM_FUNCS;
    Serial.print("function selected:");
    Serial.println(func_select);
    delay(500);
  }

  switch (func_select) {
  case 0:
      //Serial.println("Start Show_IMU()");
      Show_IMU();
      break;
  case 1:
      Go_To_Sleep();
      break;
  default:
      break;
    }

}
