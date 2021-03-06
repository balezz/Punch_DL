# Deep learning models for punch quality estimation in Boxing  

Now we have Punch Speed [android app](https://play.google.com/store/apps/details?id=ru.smart_fight.punch_speed&hl=en&gl=US) that measures punch kinematic characteristics (speed, acceleration and reaction time) with  built-in phone sensors. Also, we have a smart bracelet [Lily T-Wristband](https://aliexpress.ru/item/4000527495064.html) with ESP32 and MPU9250 on board, that much more suitable for punch measurements.  

We want to develop a system with android application and T-wristband device. This system must allows to measure punch kinematic characteristics + something more: classification type of punch and prediction punch power.  

We suppose, that Deep Learning methods will improve metrics for classification and regression tasks solving.  

### Project structure 

**PaperPunchDL** - main directory with our  paper.  

**Datasheets** - T-Wristband IMU and microcomputer datasheets.  

**Wristband_firmware** - current version of T-Wristband firmware. 
 Supports IMU reading, but have no bluetooth channel connection.
 
 ### Task list:
 - [ ] Introduction
 - [ ] Related work section
 - [ ] Methods
 - [ ] Results
 - [ ] Discussion
 - [ ] T-Wristband firmware
 - [ ] Android application
 