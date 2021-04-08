# Deep learning models for punch classification in karate  

Now we have Punch Speed [android app](https://play.google.com/store/apps/details?id=ru.smart_fight.punch_speed&hl=en&gl=US) that measures punch kinematic characteristics (speed, acceleration and reaction time) with  built-in phone sensors. Also, we have a smart bracelet [Lily T-Wristband](https://aliexpress.ru/item/4000527495064.html) with ESP32 and MPU9250 on board, that much more suitable for punch measurements.  

We want to develop a system with android application and T-wristband device. This system must allows to automatic classification type of punch.  
We suppose, that Deep Learning methods will improve metrics for classification tasks solving.  

### Project structure 

**PaperPunchDL** - main directory with our  paper.  

**Datasheets** - T-Wristband IMU and microcomputer datasheets.  

**Wristband_firmware** - current version of T-Wristband firmware. 
 Supports IMU reading, but have no bluetooth channel connection.
 
 ### Task list:
 - [x] 1 Introduction and related works 
 - [x] 2 Experimental and Methods
 - [x] 3 Results
 - [ ] 4 Discussion
 - [x] T-Wristband firmware
 - [x] Android application (bluetooth terminal for data acquisition)
 