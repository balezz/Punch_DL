
#include <pcf8563.h>
#include <TFT_eSPI.h> // Graphics and font library for ST7735 driver chip
#include <SPI.h>
#include <Wire.h>
#include <WiFi.h>
#include <time.h>
#include "sensor.h"
#include "esp_adc_cal.h"
//#include "ttgo.h"
#include "charge.h"

//  git clone -b development https://github.com/tzapu/WiFiManager.git
#include <WiFiManager.h>         //https://github.com/tzapu/WiFiManager

// #define FACTORY_HW_TEST     //! Test RTC and WiFi scan when enabled
#define ARDUINO_OTA_UPDATE      //! Enable this line OTA update


#ifdef ARDUINO_OTA_UPDATE
#include <ESPmDNS.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>
#endif


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

#define NUM_FUNCS           3 //number of different functions to switch between when pressing button
#define NUM_DEBUG           5 //number of different debug functions

extern MPU9250 IMU;

TFT_eSPI tft = TFT_eSPI();  // Invoke library, pins defined in User_Setup.h
PCF8563_Class rtc;
WiFiManager wifiManager;

char buff[256];
bool rtcIrq = false;
bool initial = 1;
bool otaStart = false;
bool otaSetup = false;
bool delayedSetup = false;
bool wifiConnected = false;
bool wifiStateChanged = false;

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

void configModeCallback (WiFiManager *myWiFiManager)
{
    Serial.println("Entered config mode");
    Serial.println(WiFi.softAPIP());
    //if you used auto generated SSID, print it
    String ssid = myWiFiManager->getConfigPortalSSID();
    Serial.println(ssid);

    tft.fillScreen(TFT_BLACK);
    tft.setTextColor(TFT_WHITE);
    tft.drawString("Connect to ",  20, tft.height() / 2 - 10);
    tft.setTextColor(TFT_GREEN);
    tft.drawString(ssid,  40, tft.height() / 2 );

}

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


void drawProgressBar(uint16_t x0, uint16_t y0, uint16_t w, uint16_t h, uint8_t percentage, uint16_t frameColor, uint16_t barColor)
{
    if (percentage == 0) {
        tft.fillRoundRect(x0, y0, w, h, 3, TFT_BLACK);
    }
    uint8_t margin = 2;
    uint16_t barHeight = h - 2 * margin;
    uint16_t barWidth = w - 2 * margin;
    tft.drawRoundRect(x0, y0, w, h, 3, frameColor);
    tft.fillRect(x0 + margin, y0 + margin, barWidth * percentage / 100.0, barHeight, barColor);
}


void setupWiFi()
{
#ifdef ARDUINO_OTA_UPDATE
    //set callback that gets called when connecting to previous WiFi fails, and enters Access Point mode
    WiFi.begin(); //try to connect to the last used network
    wifiManager.setAPCallback(configModeCallback);
    wifiManager.setBreakAfterConfig(true);          // Without this saveConfigCallback does not get fired
    wifiManager.setTimeout(120); //set a two minute timeout for wifi config
    //wifiManager.autoConnect("T-Wristband"); //TODO: switch off of auto since we can operate without wifi, provide way to get into wifi config via button press
#endif
}

void setupOTA()
{
#ifdef ARDUINO_OTA_UPDATE
    // Port defaults to 3232
    // ArduinoOTA.setPort(3232);

    // Hostname defaults to esp3232-[MAC]
    ArduinoOTA.setHostname("T-Wristband");

    // No authentication by default
    // ArduinoOTA.setPassword("admin");

    // Password can be set with it's md5 value as well
    // MD5(admin) = 21232f297a57a5a743894a0e4a801fc3
    // ArduinoOTA.setPasswordHash("21232f297a57a5a743894a0e4a801fc3");

    ArduinoOTA.onStart([]() {
        String type;
        if (ArduinoOTA.getCommand() == U_FLASH)
            type = "sketch";
        else // U_SPIFFS
            type = "filesystem";

        // NOTE: if updating SPIFFS this would be the place to unmount SPIFFS using SPIFFS.end()
        Serial.println("Start updating " + type);
        otaStart = true;
        tft.fillScreen(TFT_BLACK);
        tft.drawString("Updating...", tft.width() / 2 - 20, 55 );
    })
    .onEnd([]() {
        Serial.println("\nEnd");
        delay(500);
    })
    .onProgress([](unsigned int progress, unsigned int total) {
        // Serial.printf("Progress: %u%%\r", (progress / (total / 100)));
        int percentage = (progress / (total / 100));
        tft.setTextDatum(TC_DATUM);
        tft.setTextPadding(tft.textWidth(" 888% "));
        tft.drawString(String(percentage) + "%", 145, 35);
        drawProgressBar(10, 30, 120, 15, percentage, TFT_WHITE, TFT_BLUE);
    })
    .onError([](ota_error_t error) {
        Serial.printf("Error[%u]: ", error);
        if (error == OTA_AUTH_ERROR) Serial.println("Auth Failed");
        else if (error == OTA_BEGIN_ERROR) Serial.println("Begin Failed");
        else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed");
        else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed");
        else if (error == OTA_END_ERROR) Serial.println("End Failed");

        tft.fillScreen(TFT_BLACK);
        tft.drawString("Update Failed", tft.width() / 2 - 20, 55 );
        delay(3000);
        otaStart = false;
        initial = 1;
        targetTime = millis() + 1000;
        tft.fillScreen(TFT_BLACK);
        tft.setTextDatum(TL_DATUM);
        orig_mm = 99;
    });

    ArduinoOTA.begin();
    otaSetup = true;
#endif
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
    Serial.begin(115200);
    Serial.print("Start: ");
    Serial.println(start);
    //don't use bluetooth (for now)
    btStop();
    Serial.print("BT Stop: ");
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

    setupRTC();
    Serial.print("RTC: ");
    Serial.println(millis());

    //Serial.print("MPU: ");
    //Serial.println(millis());

    setupADC();
    Serial.print("ADC: ");
    Serial.println(millis());

    setupWiFi();
    Serial.print("WiFi: ");
    Serial.println(millis());

    //Serial.println("Setup OTA");
    //setupOTA();

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

void Show_Time()
{
    if (wifiStateChanged || initial)
    {
        //show the current ip in bottom left
        tft.setTextColor(TFT_GREEN, TFT_BLACK);
        tft.fillRect(8, 70, 80, 10, TFT_BLACK);
        tft.setCursor(8, 70);
        if (wifiConnected)
            tft.print(WiFi.localIP());
        else
            tft.print("No WiFi");
    }

    if (targetTime < millis() || initial) {
        RTC_Date datetime = rtc.getDateTime();
        hh = datetime.hour;
        mm = datetime.minute;
        ss = datetime.second;
        // Serial.printf("hh:%d mm:%d ss:%d\n", hh, mm, ss);
        targetTime = millis() + 1000;
        if (ss == 0 || initial) { //first draw or new minute
            initial = 0;
            tft.setTextColor(TFT_GREEN, TFT_BLACK);
            tft.setCursor (8, 60);
            tft.print(rtc.formatDateTime(PCF_TIMEFORMAT_YYYY_MM_DD)); // This uses the standard ADAFruit small font
        }

        tft.setTextColor(TFT_BLUE, TFT_BLACK);
        tft.drawCentreString(getVoltage(), 120, 60, 1); // Next size up font 2


        // Update digital time
        uint8_t xpos = 6;
        uint8_t ypos = 0;
        if (orig_mm != mm) { // Only redraw every minute to minimise flicker
            // Uncomment ONE of the next 2 lines, using the ghost image demonstrates text overlay as time is drawn over it
            tft.setTextColor(0x39C4, TFT_BLACK);  // Leave a 7 segment ghost image, comment out next line!
            //tft.setTextColor(TFT_BLACK, TFT_BLACK); // Set font colour to black to wipe image
            // Font 7 is to show a pseudo 7 segment display.
            // Font 7 only contains characters [space] 0 1 2 3 4 5 6 7 8 9 0 : .
            tft.drawString("88:88", xpos, ypos, 7); // Overwrite the text to clear it
            tft.setTextColor(0xFBE0, TFT_BLACK); // Orange
            orig_mm = mm;

            if (hh < 10) xpos += tft.drawChar('0', xpos, ypos, 7);
            xpos += tft.drawNumber(hh, xpos, ypos, 7);
            xcolon = xpos;
            xpos += tft.drawChar(':', xpos, ypos, 7);
            if (mm < 10) xpos += tft.drawChar('0', xpos, ypos, 7);
            tft.drawNumber(mm, xpos, ypos, 7);
        }

        if (ss % 2) { // Flash the colon
            tft.setTextColor(0x39C4, TFT_BLACK);
            xpos += tft.drawChar(':', xcolon, ypos, 7);
            tft.setTextColor(0xFBE0, TFT_BLACK);
        } else {
            tft.drawChar(':', xcolon, ypos, 7);
        }
    }
}

void Show_WiFi_Scan()
{
    if (initial)
    {
        initial = 0;
    
        tft.setTextColor(TFT_GREEN, TFT_BLACK);
        tft.fillScreen(TFT_BLACK);
        tft.setTextDatum(MC_DATUM);
        tft.setTextSize(1);

        tft.drawString("Scan Network", tft.width() / 2, tft.height() / 2);

        WiFi.mode(WIFI_STA);
        WiFi.disconnect();
        delay(100);

        int16_t n = WiFi.scanNetworks();
        tft.fillScreen(TFT_BLACK);
        if (n == 0) {
            tft.drawString("no networks found", tft.width() / 2, tft.height() / 2);
        } else {
            tft.setTextDatum(TL_DATUM);
            tft.setCursor(0, 0);
            for (int i = 0; i < n; ++i) {
                sprintf(buff,
                        "[%d]:%s(%d)",
                        i + 1,
                        WiFi.SSID(i).c_str(),
                        WiFi.RSSI(i));
                Serial.println(buff);
                tft.println(buff);
            }
        }
        WiFi.mode(WIFI_OFF);
    }
}

void Show_IMU()
{
    tft.setTextColor(TFT_GREEN, TFT_BLACK);
    tft.fillScreen(TFT_BLACK);
    tft.setTextDatum(TL_DATUM);
    readMPU9250();
    snprintf(buff, sizeof(buff), "--  ACC  GYR   MAG");
    tft.drawString(buff, 0, 0);
    snprintf(buff, sizeof(buff), "x %.2f  %.2f  %.2f", (int)1000 * IMU.ax, IMU.gx, IMU.mx);
    tft.drawString(buff, 0, 16);
    snprintf(buff, sizeof(buff), "y %.2f  %.2f  %.2f", (int)1000 * IMU.ay, IMU.gy, IMU.my);
    tft.drawString(buff, 0, 32);
    snprintf(buff, sizeof(buff), "z %.2f  %.2f  %.2f", (int)1000 * IMU.az, IMU.gz, IMU.mz);
    tft.drawString(buff, 0, 48);
    delay(200);
}

void Show_Debug_Menu()
{
    if (initial)
    {
        initial = 0;
        tft.setTextColor(TFT_YELLOW);
        //0: exit
        //1: Config WiFi
        //2: Scan
        //3: IMU debug
        //4: RTC Sync
        uint8_t yPos = 0;
        tft.drawString("Exit", 8, yPos++*16);
        tft.drawString("WiFi Scan", 8, yPos++*16);
        tft.drawString("IMU Stats", 8, yPos++*16);
        tft.drawString("WiFi Config", 8, yPos++*16);
        tft.drawString("RTC Sync", 8, yPos++*16);
        tft.drawString(">", 0, 16*debugSelect);
    }
    //tft.fillRect(0, 0, 8, 64, TFT_BLACK);
}

void Show_Debug_Item()
{
    switch (debugSelect) 
    {
        case 1:
            Show_WiFi_Scan();
            break;
        case 2:
            Show_IMU();
            break;
        case 3:
            wifiManager.startConfigPortal("T-Wristband");
            func_select = 0;
            initial = 1;
            break;
        case 4:
            Sync_RTC();
            break;
        default:
            debugSelect = 0;
            break;
    }
}

void Sync_RTC()
{
    if (initial)
    {
        initial = 0;
        //if wifi connected, check time with ntp
        if (wifiConnected)
        {
            configTime(TZ_OFFSET, DST_OFFSET, "pool.ntp.org");
            struct tm timeinfo;
            if (getLocalTime(&timeinfo))
            {
                //update rtc
                rtc.syncToRtc();
                // rtc.setDateTime((uint16_t)timeinfo.tm_year, (uint8_t)timeinfo.tm_mon, (uint8_t)timeinfo.tm_mday,
                //                 (uint8_t)timeinfo.tm_hour, (uint8_t)timeinfo.tm_min, (uint8_t)timeinfo.tm_sec);
                Serial.println("Time updated from NTP");
                tft.drawString("Time Sync Complete", 16, 36);
            }
        }
        else
        {
            tft.drawString("No WiFi Connection", 8, 36);
        }
    }
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

void Update_WiFi_State()
{
    bool connected = WiFi.isConnected();
    //Serial.println(connected);
    wifiStateChanged = (connected != wifiConnected);
    wifiConnected = connected;
    if (wifiStateChanged)
    {
        Serial.print("Wifi state changed to ");
        Serial.println(wifiConnected);
    }
}

void loop()
{
#ifdef ARDUINO_OTA_UPDATE
    if (otaSetup)
    {
        ArduinoOTA.handle();
    }
    else if (wifiConnected)
    {
        //wifi but OTA not initialized
        setupOTA();
    }
#endif

    //! If OTA starts, skip the following operation
    if (otaStart)
        return;

    if (charge_indication) {
        charge_indication = false;
        if (digitalRead(CHARGE_PIN) == LOW) {
            tft.pushImage(140, 55, 34, 16, charge);
        } else {
            tft.fillRect(140, 55, 34, 16, TFT_BLACK);
        }
    }

    Update_WiFi_State();

    if (digitalRead(TP_PIN_PIN) == HIGH) 
    {
        if (!pressed) 
        {
            pressed = true;
            pressedTime = millis();
        }
            //     tft.fillScreen(TFT_BLACK);
            //     tft.drawString("Reset WiFi Setting",  20, tft.height() / 2 );
            //     delay(3000);
            //     wifiManager.resetSettings();
            //     wifiManager.erase(true);
            //     esp_restart();
    } else {
        if (pressed)
        {
            //button released
            uint32_t duration = millis() - pressedTime;
            digitalWrite(LED_PIN, HIGH);
            delay(100);
            digitalWrite(LED_PIN, LOW);

            if (duration < 50) //too short, ignore
            { }
            else if (duration < 1000) //short press
            {
                initial = 1;
                if (func_select == 0)
                {
                    //show time
                    Go_To_Sleep();
                }
                else if (func_select == 1)
                {
                    //showing debug, select another debug item
                    debugSelect++;
                    debugSelect %= NUM_DEBUG;
                }
                else
                {
                    func_select = 0;
                }
                targetTime = millis() + 1000;
                tft.fillScreen(TFT_BLACK);
                orig_mm = 99;
            }
            else //long press
            {
                initial = 1;
                if (func_select != 1)
                {
                    debugSelect = 0;
                    func_select++;
                    func_select %= NUM_FUNCS;
                }
                else
                {
                    //select the item
                    if (debugSelect == 0)
                    { //exit back to time
                        func_select = 0;
                    }
                    else
                    { //show debug item
                        func_select = 2;
                    }
                }
                targetTime = millis() + 1000;
                tft.fillScreen(TFT_BLACK);
                orig_mm = 99;
            }
        }
        else
        {
            //it's been 30 seconds since the button was pressed, go to sleep
            if (millis() - pressedTime > 30000)
            {
                Go_To_Sleep();
            }
        }
        pressed = false;
    }

    switch (func_select) {
    case 0:
        Show_Time();
        break;
    case 1:
        //Show_WiFi_Scan();
        Show_Debug_Menu();
        break;
    case 2: 
        Show_Debug_Item();
        //Show_IMU();
        break;
    case 3:
        //Go_To_Sleep();
        break;
    default:
        break;
    }

    //Initialize items that do not need to be set up before first loop
    //Allows setup to be quicker
    if (!delayedSetup) 
    {
        delayedSetup = true;
        uint32_t start = millis();

        setupMPU9250(); //Start IMU

        Serial.print("DelayedSetup took ");
        Serial.println(millis()-start);
    }
}
